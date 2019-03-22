/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2016, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */

#include "slam-precomp.h"   // Precompiled headers



#include <mrpt/slam/CICP.h>
#include <mrpt/slam/CMetricMapBuilderICP.h>
#include <mrpt/maps/CSimplePointsMap.h>
#include <mrpt/obs/CObservationOdometry.h>
#include <mrpt/poses/CPose3DPDFGaussian.h>
#include <mrpt/poses/CPosePDFGaussian.h>
#include <mrpt/utils/CTicTac.h>
#include <mrpt/utils/CEnhancedMetaFile.h>

using namespace std;
using namespace mrpt::slam;
using namespace mrpt::obs;
using namespace mrpt::maps;
using namespace mrpt::utils;
using namespace mrpt::poses;
using namespace mrpt::math;

/*---------------------------------------------------------------
		 Constructor
  ---------------------------------------------------------------*/
CMetricMapBuilderICP::CMetricMapBuilderICP()
{
	this->initialize( CSimpleMap() );
}

/*---------------------------------------------------------------
							Destructor
  ---------------------------------------------------------------*/
CMetricMapBuilderICP::~CMetricMapBuilderICP()
{
	// Asure, we have exit all critical zones:
	enterCriticalSection();
	leaveCriticalSection();

	// Save current map to current file:
	setCurrentMapFile("");
}


/*---------------------------------------------------------------
							Options
  ---------------------------------------------------------------*/
CMetricMapBuilderICP::TConfigParams::TConfigParams() :
	matchAgainstTheGrid( false ),
	insertionLinDistance(1.0),
	insertionAngDistance(DEG2RAD(30)),
	localizationLinDistance(0.20),
	localizationAngDistance(DEG2RAD(30)),
	minICPgoodnessToAccept(0.40),
	mapInitializers()
{
}

void  CMetricMapBuilderICP::TConfigParams::loadFromConfigFile(
	const mrpt::utils::CConfigFileBase	&source,
	const std::string		&section)
{
	MRPT_LOAD_CONFIG_VAR(matchAgainstTheGrid, bool		,source,section)
	MRPT_LOAD_CONFIG_VAR(insertionLinDistance, double	,source,section)
	MRPT_LOAD_CONFIG_VAR_DEGREES(insertionAngDistance,source,section)
	MRPT_LOAD_CONFIG_VAR(localizationLinDistance, double	,source,section)
	MRPT_LOAD_CONFIG_VAR_DEGREES(localizationAngDistance, source,section)

	MRPT_LOAD_CONFIG_VAR(minICPgoodnessToAccept, double	,source,section)


	mapInitializers.loadFromConfigFile(source,section);
}

void  CMetricMapBuilderICP::TConfigParams::dumpToTextStream( CStream	&out) const
{
	mapInitializers.dumpToTextStream(out);
}

/*---------------------------------------------------------------
						processObservation
 This is the new entry point of the algorithm (the old one
  was processActionObservation, which now is a wrapper to
  this method).
  ---------------------------------------------------------------*/
void  CMetricMapBuilderICP::processObservation(const CObservationPtr &obs)
{
	mrpt::synch::CCriticalSectionLocker lock_cs( &critZoneChangingMap );

	MRPT_START

	if (metricMap.m_pointsMaps.empty() && metricMap.m_gridMaps.empty())
		throw std::runtime_error("Neither grid maps nor points map: Have you called initialize() after setting ICP_options.mapInitializers?");

	ASSERT_(obs.present())

	// Is it an odometry observation??
	if (IS_CLASS(obs,CObservationOdometry))
	{
		m_there_has_been_an_odometry = true;

		const CObservationOdometryPtr odo = CObservationOdometryPtr(obs);
		ASSERT_(odo->timestamp!=INVALID_TIMESTAMP)

		CPose2D pose_before;
		bool    pose_before_valid = m_lastPoseEst.getLatestRobotPose(pose_before);

		// Move our estimation:
		m_lastPoseEst.processUpdateNewOdometry( odo->odometry, odo->timestamp, odo->hasVelocities, odo->velocityLin, odo->velocityAng  );

		if (pose_before_valid)
		{
			// Accumulate movement:
			CPose2D pose_after;
			if (m_lastPoseEst.getLatestRobotPose(pose_after))
				this->accumulateRobotDisplacementCounters(pose_after);
		}
	} // end it's odometry
	else
	{
		// Current robot pose given the timestamp of the observation (this can include a small extrapolation
		//  using the latest known robot velocities):
		CPose2D		initialEstimatedRobotPose;
		{
			float v,w;
			if (obs->timestamp!=INVALID_TIMESTAMP)
			{
				if (!m_lastPoseEst.getCurrentEstimate(initialEstimatedRobotPose,v,w, obs->timestamp))
				{	// couldn't had a good extrapolation estimate... we'll have to live with the latest pose:
					m_lastPoseEst.getLatestRobotPose(initialEstimatedRobotPose);
				}
			}
			else
				m_lastPoseEst.getLatestRobotPose(initialEstimatedRobotPose);
		}

		// To know the total path length:
		CPose2D  previousKnownRobotPose;
		m_lastPoseEst.getLatestRobotPose(previousKnownRobotPose);

		// Increment (this may only include the effects of extrapolation with velocity...):
		this->accumulateRobotDisplacementCounters(previousKnownRobotPose);  // initialEstimatedRobotPose-previousKnownRobotPose);

		// We'll skip ICP-based localization for this observation only if:
		//  - We had some odometry since the last pose correction (m_there_has_been_an_odometry=true).
		//  - AND, the traversed distance is small enough:
		const bool we_skip_ICP_pose_correction =
			m_there_has_been_an_odometry &&
			m_distSinceLastICP.lin < std::min(ICP_options.localizationLinDistance,ICP_options.insertionLinDistance) &&
			m_distSinceLastICP.ang < std::min(ICP_options.localizationAngDistance,ICP_options.insertionAngDistance);

		CICP::TReturnInfo	icpReturn;
		bool				can_do_icp=false;

		// Select the map to match with ....
		CMetricMap   *matchWith = NULL;
		if (ICP_options.matchAgainstTheGrid && !metricMap.m_gridMaps.empty() )
		{
			matchWith = static_cast<CMetricMap*>(metricMap.m_gridMaps[0].pointer());
		}
		else
		{
			ASSERTMSG_( metricMap.m_pointsMaps.size(), "No points map in multi-metric map." )
			matchWith = static_cast<CMetricMap*>(metricMap.m_pointsMaps[0].pointer());
		}
		ASSERT_(matchWith!=NULL)

		if (we_skip_ICP_pose_correction)
		{
			if (options.verbose)
				printf("[CMetricMapBuilderICP] Skipping ICP pose correction...\n" );
		}
		else
		{
			m_there_has_been_an_odometry = false;

			// --------------------------------------------------------------------------------------
			// Any other observation:
			//  1) If the observation generates points in a point map, do ICP
			//  2) In any case, insert the observation if the minimum distance has been satisfaced.
			// --------------------------------------------------------------------------------------
			CSimplePointsMap   sensedPoints;
			sensedPoints.insertionOptions.minDistBetweenLaserPoints = 0.02f;
			sensedPoints.insertionOptions.also_interpolate = false;

			// Create points representation of the observation:
			// Insert only those planar range scans in the altitude of the grid map:
			if (ICP_options.matchAgainstTheGrid &&
				!metricMap.m_gridMaps.empty() &&
				metricMap.m_gridMaps[0]->insertionOptions.useMapAltitude)
			{
				// Use grid altitude:
				if (IS_CLASS(obs,CObservation2DRangeScan ) )
				{
					CObservation2DRangeScanPtr obsLaser = CObservation2DRangeScanPtr(obs);
					if ( std::abs( metricMap.m_gridMaps[0]->insertionOptions.mapAltitude - obsLaser->sensorPose.z())<0.01)
						can_do_icp = sensedPoints.insertObservationPtr(obs);
				}
			}
			else
			{
				// Do not use grid altitude:
				can_do_icp = sensedPoints.insertObservationPtr(obs);
			}

			if (IS_DERIVED(matchWith,CPointsMap) && static_cast<CPointsMap*>(matchWith)->empty())
				can_do_icp = false;	// The reference map is empty!

			if (can_do_icp)
			{
				// We DO HAVE points with this observation:
				// Execute ICP over the current points map and the sensed points:
				// ----------------------------------------------------------------------
				CICP	ICP;
				float	runningTime;

				ICP.options = ICP_params;

				CPosePDFPtr pestPose= ICP.Align(
					matchWith,					// Map 1
					&sensedPoints,				// Map 2
					initialEstimatedRobotPose,	// a first gross estimation of map 2 relative to map 1.
					&runningTime,				// Running time
					&icpReturn					// Returned information
					);

				if (icpReturn.goodness> ICP_options.minICPgoodnessToAccept)
				{
					// save estimation:
					CPosePDFGaussian  pEst2D;
					pEst2D.copyFrom( *pestPose );

					m_lastPoseEst.processUpdateNewPoseLocalization( TPose2D(pEst2D.mean), pEst2D.cov, obs->timestamp );
					m_lastPoseEst_cov = pEst2D.cov;

					m_distSinceLastICP.updatePose(pEst2D.mean);


					// Debug output to console:
					if (options.verbose)
					{
						cout << "[CMetricMapBuilderICP]  " << previousKnownRobotPose << "->" << pEst2D.getMeanVal() << std::endl;
						cout << format("[CMetricMapBuilderICP]   Fit:%.1f%% Itr:%i In %.02fms \n",
							icpReturn.goodness*100,
							icpReturn.nIterations,
							1000*runningTime );
					}
				}
				else
				{
					if (options.verbose)
						cout << "[CMetricMapBuilderICP]  Ignoring ICP of low quality: " << icpReturn.goodness*100 << std::endl;
				}

				// Compute the transversed length:
				CPose2D  currentKnownRobotPose;
				m_lastPoseEst.getLatestRobotPose(currentKnownRobotPose);

				this->accumulateRobotDisplacementCounters(currentKnownRobotPose); //currentKnownRobotPose - previousKnownRobotPose);

			} // end we can do ICP.

		} // else, we do ICP pose correction


		// ----------------------------------------------------------
		//				CRITERION TO DECIDE MAP UPDATE:
		//   A distance large-enough from the last update for each sensor, AND
		//    either: (i) this was a good match or (ii) this is the first time for this sensor.
		// ----------------------------------------------------------
		const bool firstTimeForThisSensor = m_distSinceLastInsertion.find(obs->sensorLabel)==m_distSinceLastInsertion.end();
		bool update = 	firstTimeForThisSensor || 
						( (!can_do_icp || icpReturn.goodness>ICP_options.minICPgoodnessToAccept) &&
						( m_distSinceLastInsertion[obs->sensorLabel].lin >= ICP_options.insertionLinDistance ||
						  m_distSinceLastInsertion[obs->sensorLabel].ang >= ICP_options.insertionAngDistance ) );

		// Used any "options.alwaysInsertByClass" ??
		if (options.alwaysInsertByClass.contains(obs->GetRuntimeClass()))
			update = true;

		// We need to always insert ALL the observations at the beginning until the first one
		//  that actually insert some points into the map used as a reference, since otherwise
		//  we'll not be able to do ICP against an empty map!!
		if (matchWith && matchWith->isEmpty())
			update = true;

		// ----------------------------------------------------------
		// ----------------------------------------------------------
		if ( options.enableMapUpdating && update)
		{
			CTicTac tictac;

			if (options.verbose)
				tictac.Tic();

			// Insert the observation:
			CPose2D  currentKnownRobotPose;
			m_lastPoseEst.getLatestRobotPose(currentKnownRobotPose);

			// Create new entry:
			m_distSinceLastInsertion[obs->sensorLabel].last_update = currentKnownRobotPose;

			// Reset distance counters:
			resetRobotDisplacementCounters(currentKnownRobotPose);
			//m_distSinceLastInsertion[obs->sensorLabel].updatePose(currentKnownRobotPose);

			if (options.verbose)
				printf("[CMetricMapBuilderICP] Updating map from pose %s\n",currentKnownRobotPose.asString().c_str());

			CPose3D		estimatedPose3D(currentKnownRobotPose);
			metricMap.insertObservationPtr(obs,&estimatedPose3D);

			// Add to the vector of "poses"-"SFs" pairs:
			CPosePDFGaussian	posePDF(currentKnownRobotPose);
			CPose3DPDFPtr  pose3D = CPose3DPDFPtr( CPose3DPDF::createFrom2D( posePDF ) );

			CSensoryFramePtr sf = CSensoryFrame::Create();
			sf->insert(obs);

			SF_Poses_seq.insert( pose3D, sf );

			if (options.verbose)
				printf("[CMetricMapBuilderICP] Map updated OK!! In %.03fms\n",tictac.Tac()*1000.0f );
		}



	} // end other observation

	// Robot path history:
	{
		TPose2D p;
		if (m_lastPoseEst.getLatestRobotPose(p))
			m_estRobotPath.push_back(p);
	}

	MRPT_END

} // end processObservation

/*---------------------------------------------------------------

						processActionObservation

  ---------------------------------------------------------------*/
void  CMetricMapBuilderICP::processActionObservation(
			CActionCollection	&action,
			CSensoryFrame		&in_SF )
{
	// 1) process action:
	CActionRobotMovement2DPtr movEstimation = action.getBestMovementEstimation();
	if (movEstimation)
	{
		m_auxAccumOdometry.composeFrom( m_auxAccumOdometry, movEstimation->poseChange->getMeanVal() );

		CObservationOdometryPtr obs = CObservationOdometry::Create();
		obs->timestamp = movEstimation->timestamp;
		obs->odometry = m_auxAccumOdometry;
		this->processObservation(obs);
	}

	// 2) Process observations one by one:
	for (CSensoryFrame::iterator i=in_SF.begin();i!=in_SF.end();++i)
		this->processObservation(*i);

}

/*---------------------------------------------------------------
						setCurrentMapFile
  ---------------------------------------------------------------*/
void  CMetricMapBuilderICP::setCurrentMapFile( const char *mapFile )
{
	// Save current map to current file:
	if (currentMapFile.size())
		saveCurrentMapToFile( currentMapFile.c_str() );

	// Sets new current map file:
	currentMapFile = mapFile;

	// Load map from file or create an empty one:
	if (currentMapFile.size())
		loadCurrentMapFromFile( mapFile );
}


/*---------------------------------------------------------------
						getCurrentPoseEstimation
  ---------------------------------------------------------------*/
CPose3DPDFPtr CMetricMapBuilderICP::getCurrentPoseEstimation() const
{
	CPosePDFGaussian  pdf2D;
	m_lastPoseEst.getLatestRobotPose(pdf2D.mean);
	pdf2D.cov = m_lastPoseEst_cov;

	CPose3DPDFGaussianPtr pdf3D = CPose3DPDFGaussian::Create();
	pdf3D->copyFrom(pdf2D);
	return pdf3D;
}

/*---------------------------------------------------------------
						initialize
  ---------------------------------------------------------------*/
void  CMetricMapBuilderICP::initialize(
	const CSimpleMap  &initialMap,
	CPosePDF					*x0 )
{
	MRPT_START

	// Reset vars:
	m_estRobotPath.clear();
	m_auxAccumOdometry = CPose2D(0,0,0);

	m_distSinceLastICP.lin = m_distSinceLastICP.ang = 0;
	m_distSinceLastInsertion.clear();

	m_there_has_been_an_odometry = false;

	// Init path & map:
	mrpt::synch::CCriticalSectionLocker lock_cs( &critZoneChangingMap );

	// Create metric maps:
	metricMap.setListOfMaps( &ICP_options.mapInitializers );

	// copy map:
	SF_Poses_seq = initialMap;

	// Parse SFs to the hybrid map:
	// Set options:
	// ---------------------
	//if (metricMap.m_pointsMaps.size())
	//{
	//	metricMap.m_pointsMaps[0]->insertionOptions.fuseWithExisting			= false;
	//	metricMap.m_pointsMaps[0]->insertionOptions.minDistBetweenLaserPoints = 0.05f;
	//	metricMap.m_pointsMaps[0]->insertionOptions.disableDeletion			= true;
	//	metricMap.m_pointsMaps[0]->insertionOptions.isPlanarMap				= true;
	//	metricMap.m_pointsMaps[0]->insertionOptions.matchStaticPointsOnly		= true;
	//}

	// Load estimated pose from given PDF:
	m_lastPoseEst.reset();

	if (x0)
		m_lastPoseEst.processUpdateNewPoseLocalization( x0->getMeanVal(), CMatrixDouble33(), mrpt::system::now() );

	for (size_t i=0;i<SF_Poses_seq.size();i++)
	{
		CPose3DPDFPtr		posePDF;
		CSensoryFramePtr	SF;

		// Get the SF and its pose:
		SF_Poses_seq.get(i, posePDF,SF);

		CPose3D	 estimatedPose3D;
		posePDF->getMean(estimatedPose3D);

		// Insert observations into the map:
		SF->insertObservationsInto( &metricMap, &estimatedPose3D );
	}

	if (options.verbose)
		printf("[CMetricMapBuilderICP::loadCurrentMapFromFile] OK\n");

	MRPT_END
}

/*---------------------------------------------------------------
						getCurrentMapPoints
  ---------------------------------------------------------------*/
void  CMetricMapBuilderICP::getCurrentMapPoints(
	std::vector<float>		&x,
	std::vector<float>		&y)
{
	// Critical section: We are using our global metric map
	enterCriticalSection();

	ASSERT_( metricMap.m_pointsMaps.size()>0 );
	metricMap.m_pointsMaps[0]->getAllPoints(x,y);

	// Exit critical zone.
	leaveCriticalSection();
}

/*---------------------------------------------------------------
				getCurrentlyBuiltMap
  ---------------------------------------------------------------*/
void  CMetricMapBuilderICP::getCurrentlyBuiltMap(CSimpleMap &out_map) const
{
	out_map = SF_Poses_seq;

}

/*---------------------------------------------------------------
						getCurrentlyBuiltMetricMap
  ---------------------------------------------------------------*/
CMultiMetricMap*   CMetricMapBuilderICP::getCurrentlyBuiltMetricMap()
{
	return &metricMap;
}


/*---------------------------------------------------------------
			getCurrentlyBuiltMapSize
  ---------------------------------------------------------------*/
unsigned int  CMetricMapBuilderICP::getCurrentlyBuiltMapSize()
{
	return SF_Poses_seq.size();
}

/*---------------------------------------------------------------
				saveCurrentEstimationToImage
  ---------------------------------------------------------------*/
void  CMetricMapBuilderICP::saveCurrentEstimationToImage(const std::string &file, bool formatEMF_BMP )
{
	MRPT_START

	CImage        img;
	const size_t  nPoses = m_estRobotPath.size();

	ASSERT_( metricMap.m_gridMaps.size()>0 );

	if (!formatEMF_BMP)
		THROW_EXCEPTION("Not implemented yet for BMP!");

	// grid map as bitmap:
	// ----------------------------------
	metricMap.m_gridMaps[0]->getAsImage( img );

	// Draw paths (using vectorial plots!) over the EMF file:
	// -------------------------------------------------
//	float					SCALE = 1000;
	CEnhancedMetaFile		EMF( file, 1000 );

	EMF.drawImage( 0,0, img );

	unsigned int imgHeight = img.getHeight();

	// Path hypothesis:
	// ----------------------------------
	int		x1,x2,y1,y2;

	// First point: (0,0)
	x2 = metricMap.m_gridMaps[0]->x2idx( 0.0f );
	y2 = metricMap.m_gridMaps[0]->y2idx( 0.0f );

	// Draw path in the bitmap:
	for (size_t j=0;j<nPoses;j++)
	{
		// For next segment
		x1 = x2;
		y1 = y2;

		// Coordinates -> pixels
		x2 = metricMap.m_gridMaps[0]->x2idx( m_estRobotPath[j].x );
		y2 = metricMap.m_gridMaps[0]->y2idx( m_estRobotPath[j].y );

		// Draw line:
		EMF.line(
			x1, imgHeight-1-y1,
			x2, imgHeight-1-y2,
			TColor::black );
	}

	MRPT_END
}


void CMetricMapBuilderICP::accumulateRobotDisplacementCounters(const CPose2D &new_pose)
{
	m_distSinceLastICP.updateDistances(new_pose);

	for (mrpt::aligned_containers<std::string,CMetricMapBuilderICP::TDist>::map_t::iterator  it=m_distSinceLastInsertion.begin();it!=m_distSinceLastInsertion.end();++it)
		it->second.updateDistances(new_pose);
}

void CMetricMapBuilderICP::resetRobotDisplacementCounters(const CPose2D &new_pose)
{
	m_distSinceLastICP.updatePose(new_pose);

	for (mrpt::aligned_containers<std::string,CMetricMapBuilderICP::TDist>::map_t::iterator  it=m_distSinceLastInsertion.begin();it!=m_distSinceLastInsertion.end();++it)
		it->second.updatePose(new_pose);
}

void CMetricMapBuilderICP::TDist::updateDistances(const mrpt::poses::CPose2D &p)
{
	mrpt::poses::CPose2D Ap = p - this->last_update;
	lin = Ap.norm();
	ang = std::abs( Ap.phi() );
}

void CMetricMapBuilderICP::TDist::updatePose(const mrpt::poses::CPose2D &p)
{
	this->last_update = p; 
	lin = 0;
	ang = 0;
}

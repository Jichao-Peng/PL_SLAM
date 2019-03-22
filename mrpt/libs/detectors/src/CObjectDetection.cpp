/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2016, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */

#include "detectors-precomp.h"  // Precompiled headers

#include <mrpt/detectors/CObjectDetection.h>
#include <mrpt/obs/CObservationImage.h>
#include <mrpt/utils/CStartUpClassesRegister.h>
#include <mrpt/obs/CObservationImage.h>
#include <mrpt/obs/CObservationStereoImages.h>
#include <mrpt/obs/CObservation3DRangeScan.h>

// Universal include for all versions of OpenCV
#include <mrpt/otherlibs/do_opencv_includes.h> 

using namespace mrpt::detectors;
using namespace mrpt::utils;

extern CStartUpClassesRegister  mrpt_detectors_class_reg;

void CObjectDetection::detectObjects(const CImage *img, vector_detectable_object &detected)
{
	//static const int dumm2 =
	mrpt_detectors_class_reg.do_nothing(); // Avoid compiler removing this class in static linking

	mrpt::obs::CObservationImage o;
	o.timestamp = mrpt::system::now();
	o.image.setFromImageReadOnly(*img);
	this->detectObjects_Impl(&o,detected);
}

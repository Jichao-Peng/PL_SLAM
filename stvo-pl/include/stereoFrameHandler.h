/*****************************************************************************
**      Stereo VO and SLAM by combining point and line segment features     **
******************************************************************************
**                                                                          **
**  Copyright(c) 2016-2018, Ruben Gomez-Ojeda, University of Malaga         **
**  Copyright(c) 2016-2018, David Zuñiga-Noël, University of Malaga         **
**  Copyright(c) 2016-2018, MAPIR group, University of Malaga               **
**                                                                          **
**  This program is free software: you can redistribute it and/or modify    **
**  it under the terms of the GNU General Public License (version 3) as     **
**  published by the Free Software Foundation.                              **
**                                                                          **
**  This program is distributed in the hope that it will be useful, but     **
**  WITHOUT ANY WARRANTY; without even the implied warranty of              **
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the            **
**  GNU General Public License for more details.                            **
**                                                                          **
**  You should have received a copy of the GNU General Public License       **
**  along with this program.  If not, see <http://www.gnu.org/licenses/>.   **
**                                                                          **
*****************************************************************************/

#pragma once
#include <stereoFrame.h>
#include <stereoFeatures.h>
#include "auxiliar.h"

typedef Matrix<double,6,6> Matrix6d;
typedef Matrix<double,6,1> Vector6d;

class StereoFrame;

namespace StVO{

class StereoFrameHandler
{

public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    StereoFrameHandler( PinholeStereoCamera* cam_ );
    ~StereoFrameHandler();

    void initialize( const Mat img_l_, const Mat img_r_, const int idx_);
    void insertStereoPair(const Mat img_l_, const Mat img_r_, const int idx_);
    void updateFrame();

    void f2fTracking();
    void matchF2FPoints();
    void matchF2FLines();
    double f2fLineSegmentOverlap( Vector2d spl_obs, Vector2d epl_obs, Vector2d spl_proj, Vector2d epl_proj  );

    bool isGoodSolution( Matrix4d DT, Matrix6d DTcov, double err );
    void optimizePose();
    void resetOutliers();
    void setAsOutliers();

    void plotStereoFrameProjerr(Matrix4d DT, int iter);
    void plotLeftPair();

    // adaptative fast
    int orb_fast_th;    //orb特征提取FAST的threshold（阈值）
    double llength_th;  

    // slam-specific functions
    bool needNewKF();
    void currFrameIsKF();
    
    /*求Hx,Hp*/
    
    Matrix26d getPointHx(Vector6d x,Vector3d Pc);
    Matrix23d getPointHp(Vector6d x,Vector3d Pc);


    //list< boost::shared_ptr<PointFeature> > matched_pt;
    //list< boost::shared_ptr<LineFeature>  > matched_ls;

    list< PointFeature* > matched_pt;   //匹配的特征点集合 
    list< LineFeature*  > matched_ls;   //匹配的线特征集合

    StereoFrame* prev_frame;            //前一帧
    StereoFrame* curr_frame;            //当前帧
    PinholeStereoCamera *cam;           //针孔相机模型

    int  n_inliers, n_inliers_pt, n_inliers_ls;     //内点数量

    // slam-specific variables
    bool     prev_f_iskf;                   //一个flag，前一帧是否为关键帧
    double   entropy_first_prevKF;          //熵，论文中提到，将代表不确定性的协方差矩阵转换为一个标量，称之为entropy
    Matrix4d T_prevKF;                      //前一个关键帧的位姿
    Matrix4d T_w_curr;                      //当前帧到世界坐标系
    Matrix6d cov_prevKF_currF;
    int      N_prevKF_currF;

//    bool recurse;

private:

    void prefilterOutliers( Matrix4d DT );
    // good point features selection
    void gfPointSeclet_Greedy(Matrix4d DT);
    void gfPointSeclet(Matrix4d DT);
    void removeOutliers( Matrix4d DT );
    void gaussNewtonOptimization(Matrix4d &DT, Matrix6d &DT_cov, double &err_, int max_iters);
    void gaussNewtonOptimizationRobust(Matrix4d &DT, Matrix6d &DT_cov, double &err_, int max_iters);
    void gaussNewtonOptimizationRobustDebug(Matrix4d &DT, Matrix6d &DT_cov, double &err_, int max_iters);
    void levenbergMarquardtOptimization(Matrix4d &DT, Matrix6d &DT_cov, double &err_, int max_iters);
    void optimizeFunctions(Matrix4d DT, Matrix6d &H, Vector6d &g, double &e);
    void optimizeFunctionsRobust(Matrix4d DT, Matrix6d &H, Vector6d &g, double &e);
    void optimizePoseDebug();
    void getLineJacobi(const LineFeature* line,Matrix4d DT,Vector2d& err_i,Vector6d& J_aux);
    void getPoseInfoOnLine(const LineFeature * line,Matrix<double, 6, 6> & info_pose);
};

}

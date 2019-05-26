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

#include <stereoFrameHandler.h>
#include <random>
#include "matching.h"
#include "stereoFeatures.h"
#include "timer.h"

namespace StVO{
StereoFrameHandler::StereoFrameHandler( PinholeStereoCamera *cam_ ) : cam(cam_) {}

StereoFrameHandler::~StereoFrameHandler(){}

/*  main methods  */
// 初始化函数，左右图像，帧数索引
//就是把左右相机的首帧作为前一帧，并把它们的位姿初始化为单位矩阵。同时从config中获取ORB提取特征的threshold。
void StereoFrameHandler::initialize(const Mat img_l_, const Mat img_r_ , const int idx_)
{
    // variables for adaptative thresholds
    orb_fast_th = Config::orbFastTh();//初始化orb_fast_th
    llength_th  = Config::minLineLength() * std::min( cam->getWidth(), cam->getHeight() ) ; // 确定直线的最短距离
    // define StereoFrame
    //新建双目帧 左右图像，序列号，相机模型，然后建立像素和栅格的转换关系
    prev_frame = new StereoFrame( img_l_, img_r_, idx_, cam ); 
    //提取点特征和线特征
    prev_frame->extractStereoFeatures( llength_th, orb_fast_th );
    //位姿是单位矩阵
    prev_frame->T_kf_f     = Matrix4d::Identity();
    prev_frame->Tkf_f_cov = Matrix6d::Identity();
    prev_frame->DT      = Matrix4d::Identity();
    curr_frame = prev_frame;
    // SLAM variables for KF decision  关键帧选择的量
    T_prevKF         = Matrix4d::Identity();
    cov_prevKF_currF = Matrix6d::Zero();
    prev_f_iskf      = true;
    N_prevKF_currF   = 0;
}

//更新特征以及匹配关系l
void StereoFrameHandler::insertStereoPair(const Mat img_l_, const Mat img_r_ , const int idx_)
{
    //curr_frame.reset( new StereoFrame( img_l_, img_r_, idx_, cam ) );
    curr_frame =  new StereoFrame( img_l_, img_r_, idx_, cam );
    curr_frame->extractStereoFeatures( llength_th, orb_fast_th ); //提取特征
    f2fTracking(); //帧间跟踪 （并不计算位姿，只是建立匹配关系）
}

//更新orb_fast_th，将当前帧作为上一帧，清除当前帧
void StereoFrameHandler::updateFrame()
{

    // update FAST threshold for the keypoint detection
    if( Config::adaptativeFAST() )
    {
        int min_fast  = Config::fastMinTh();
        int max_fast  = Config::fastMaxTh();
        int fast_inc  = Config::fastIncTh();
        int feat_th   = Config::fastFeatTh();
        float err_th  = Config::fastErrTh();

        // if bad optimization, -= 2*fast_inc
        if( curr_frame->DT == Matrix4d::Identity() || curr_frame->err_norm > err_th )
            orb_fast_th = std::max( min_fast, orb_fast_th - 2*fast_inc );
        // elif number of features ...
        else if( n_inliers_pt < feat_th )
            orb_fast_th = std::max( min_fast, orb_fast_th - 2*fast_inc );
        else if( n_inliers_pt < feat_th * 2 )
            orb_fast_th = std::max( min_fast, orb_fast_th - fast_inc );
        else if( n_inliers_pt > feat_th * 3 )
            orb_fast_th = std::min( max_fast, orb_fast_th + fast_inc );
        else if( n_inliers_pt > feat_th * 4 )
            orb_fast_th = std::min( max_fast, orb_fast_th + 2*fast_inc );
    }

    // clean and update variables
    for( auto pt: matched_pt )
        delete pt;
    for( auto ls: matched_ls )
        delete ls;
    matched_pt.clear();
    matched_ls.clear();

    //prev_frame.reset();
    //prev_frame.reset( curr_frame );
    delete prev_frame;
    prev_frame = curr_frame;
    curr_frame = NULL;

}

/*  tracking methods  */
//更新matched_pt，内点数量，当前帧和上一帧的匹配关系和部分成员变量，如pl_obs，idx，inlier
void StereoFrameHandler::f2fTracking()
{
    Timer timer;
    // feature matching 这个是值两帧之间的匹配，不是左右图像
    matched_pt.clear();
    matched_ls.clear();

    //判断是否有点，有线，是否共同使用
    if( Config::plInParallel() && Config::hasPoints() && Config::hasLines() )
    {
        timer.start();
        auto detect_p = async(launch::async, &StereoFrameHandler::matchF2FPoints, this );
        auto detect_l = async(launch::async, &StereoFrameHandler::matchF2FLines,  this );
        detect_p.wait();
        detect_l.wait();
        //cout << endl << "VO matchF2FLinesPoints: " << timer.stop() << endl;
    }
    else
    {
        if (Config::hasPoints()) matchF2FPoints();
        if (Config::hasLines()) matchF2FLines();
    }
    //最后内点的数量是点特征内点数量+线特征内点数量
    n_inliers_pt = matched_pt.size();
    n_inliers_ls = matched_ls.size();
    n_inliers    = n_inliers_pt + n_inliers_ls;
}
/*
①匹配描述子时，使用暴力匹配，Hamming距离；
②分别计算前一帧到当前帧的匹配 pmatches_12，和当前帧到前一帧的匹配 pmatches_21；
③把以上两种匹配距离都以描述子距离从小到大的的方式排序；
④然后，遍历匹配 pmatches_12，如果 pmatches_12 的询问点和  pmatches_21的训练点是一样的，并且 pmatches_12 的最佳匹配距离比次优匹配的距离的Config::minRatio12P()还小，则认为这个点特征是内点，并把该点放到匹配点集 matched_pt中。
当然，用不用这种互匹配方式来选择内点，可以根据参数 Config::bestLRMatches() 来选择。
 */
void StereoFrameHandler::matchF2FPoints()
{

    // points f2f tracking
    // --------------------------------------------------------------------------------------------------------------------
    matched_pt.clear();
    if ( !Config::hasPoints() || curr_frame->stereo_pt.empty() || prev_frame->stereo_pt.empty() )
        return;

    std::vector<int> matches_12;
    //min_ratio_12_p    : 0.75       # min. ratio between the first and second best matches
    //获得 matches_12 上一帧所有特征点对应的当前帧的匹配点的索引集合
    ////不用坐标信息吗？还是已经包含了坐标信息
    match(prev_frame->pdesc_l, curr_frame->pdesc_l, Config::minRatio12P(), matches_12);

    // bucle around pmatches
    for (int i1 = 0; i1 < matches_12.size(); ++i1) {
        //i2是当前帧对应上一帧i1的匹配特征的索引
        const int i2 = matches_12[i1];
        if (i2 < 0) continue;//i2=-1
        /*执行匹配过程，并更新观测，更新内点*/
        prev_frame->stereo_pt[i1]->pl_obs = curr_frame->stereo_pt[i2]->pl;
        prev_frame->stereo_pt[i1]->P_obs = curr_frame->stereo_pt[i2]->P;
        prev_frame->stereo_pt[i1]->inlier = true;
        matched_pt.push_back( prev_frame->stereo_pt[i1]->safeCopy() );
        //以先前帧为标准，当前帧和上一帧的匹配点索引相同
        curr_frame->stereo_pt[i2]->idx = prev_frame->stereo_pt[i1]->idx; // prev idx
    }
}
/*
①提取线特征方法：LSD——Line Segment Detector，具有高精确性和可重复性；
②线特征描述子：用LBD——Line Band Descriptor方法，原文中是说：allows us to find correspondences between lines based on their local appearance。
③确定内点的方法和点特征的类似，互相匹配检测，以及这个线特征是否有足够的意义；
④利用线段提供的有用的几何信息来过滤出不同方向和长度的线条。
 */
void StereoFrameHandler::matchF2FLines()
{

    // line segments f2f tracking
    matched_ls.clear();
    if( !Config::hasLines() || curr_frame->stereo_ls.empty() || prev_frame->stereo_ls.empty() )
        return;

    std::vector<int> matches_12;
    //不用坐标信息吗？还是已经包含了坐标信息
    match(prev_frame->ldesc_l, curr_frame->ldesc_l, Config::minRatio12L(), matches_12);

    // bucle around pmatches
    for (int i1 = 0; i1 < matches_12.size(); ++i1) {
        const int i2 = matches_12[i1];
        if (i2 < 0) continue;
        prev_frame->stereo_ls[i1]->sdisp_obs = curr_frame->stereo_ls[i2]->sdisp;
        prev_frame->stereo_ls[i1]->edisp_obs = curr_frame->stereo_ls[i2]->edisp;
        prev_frame->stereo_ls[i1]->spl_obs   = curr_frame->stereo_ls[i2]->spl;
        prev_frame->stereo_ls[i1]->epl_obs   = curr_frame->stereo_ls[i2]->epl;
        prev_frame->stereo_ls[i1]->le_obs    = curr_frame->stereo_ls[i2]->le;
        prev_frame->stereo_ls[i1]->plucCam_obs = curr_frame->stereo_ls[i2]->plucCam;
        prev_frame->stereo_ls[i1]->inlier    = true;
        matched_ls.push_back( prev_frame->stereo_ls[i1]->safeCopy() );
        curr_frame->stereo_ls[i2]->idx = prev_frame->stereo_ls[i1]->idx; // prev idx
    }
}

//计算线特征的观测线段和投影线段的重合部分比率
double StereoFrameHandler::f2fLineSegmentOverlap( Vector2d spl_obs, Vector2d epl_obs, Vector2d spl_proj, Vector2d epl_proj  )
{

    double overlap = 1.f;

    if( std::abs(spl_obs(0)-epl_obs(0)) < 1.0 )         // vertical lines
    {

        // line equations
        Vector2d l = epl_obs - spl_obs;

        // intersection points
        Vector2d spl_proj_line, epl_proj_line;
        spl_proj_line << spl_obs(0), spl_proj(1);
        epl_proj_line << epl_obs(0), epl_proj(1);

        // estimate overlap in function of lambdas
        double lambda_s = (spl_proj_line(1)-spl_obs(1)) / l(1);
        double lambda_e = (epl_proj_line(1)-spl_obs(1)) / l(1);

        double lambda_min = min(lambda_s,lambda_e);
        double lambda_max = max(lambda_s,lambda_e);

        if( lambda_min < 0.f && lambda_max > 1.f )
            overlap = 1.f;
        else if( lambda_max < 0.f || lambda_min > 1.f )
            overlap = 0.f;
        else if( lambda_min < 0.f )
            overlap = lambda_max;
        else if( lambda_max > 1.f )
            overlap = 1.f - lambda_min;
        else
            overlap = lambda_max - lambda_min;

    }
    else if( std::abs(spl_obs(1)-epl_obs(1)) < 1.0 )    // horizontal lines (previously removed)
    {

        // line equations
        Vector2d l = epl_obs - spl_obs;

        // intersection points
        Vector2d spl_proj_line, epl_proj_line;
        spl_proj_line << spl_proj(0), spl_obs(1);
        epl_proj_line << epl_proj(0), epl_obs(1);

        // estimate overlap in function of lambdas
        double lambda_s = (spl_proj_line(0)-spl_obs(0)) / l(0);
        double lambda_e = (epl_proj_line(0)-spl_obs(0)) / l(0);

        double lambda_min = min(lambda_s,lambda_e);
        double lambda_max = max(lambda_s,lambda_e);

        if( lambda_min < 0.f && lambda_max > 1.f )
            overlap = 1.f;
        else if( lambda_max < 0.f || lambda_min > 1.f )
            overlap = 0.f;
        else if( lambda_min < 0.f )
            overlap = lambda_max;
        else if( lambda_max > 1.f )
            overlap = 1.f - lambda_min;
        else
            overlap = lambda_max - lambda_min;

    }
    else                                            // non-degenerate cases
    {

        // line equations
        Vector2d l = epl_obs - spl_obs;
        double a = spl_obs(1)-epl_obs(1);
        double b = epl_obs(0)-spl_obs(0);
        double c = spl_obs(0)*epl_obs(1) - epl_obs(0)*spl_obs(1);

        // intersection points
        Vector2d spl_proj_line, epl_proj_line;
        double lxy = 1.f / (a*a+b*b);

        spl_proj_line << ( b*( b*spl_proj(0)-a*spl_proj(1))-a*c ) * lxy,
                         ( a*(-b*spl_proj(0)+a*spl_proj(1))-b*c ) * lxy;

        epl_proj_line << ( b*( b*epl_proj(0)-a*epl_proj(1))-a*c ) * lxy,
                         ( a*(-b*epl_proj(0)+a*epl_proj(1))-b*c ) * lxy;

        // estimate overlap in function of lambdas
        double lambda_s = (spl_proj_line(0)-spl_obs(0)) / l(0);
        double lambda_e = (epl_proj_line(0)-spl_obs(0)) / l(0);

        double lambda_min = min(lambda_s,lambda_e);
        double lambda_max = max(lambda_s,lambda_e);

        if( lambda_min < 0.f && lambda_max > 1.f )
            overlap = 1.f;
        else if( lambda_max < 0.f || lambda_min > 1.f )
            overlap = 0.f;
        else if( lambda_min < 0.f )
            overlap = lambda_max;
        else if( lambda_max > 1.f )
            overlap = 1.f - lambda_min;
        else
            overlap = lambda_max - lambda_min;

    }

    return overlap;

}

/*  optimization functions */

bool StereoFrameHandler::isGoodSolution( Matrix4d DT, Matrix6d DTcov, double err )
{
    SelfAdjointEigenSolver<Matrix6d> eigensolver(DTcov);
    Vector6d DT_cov_eig = eigensolver.eigenvalues();

    //特征值大于1或者小于0意味着什么？
    if( DT_cov_eig(0)<0.0 || DT_cov_eig(5)>1.0 || err < 0.0 || err > 1.0 || !is_finite(DT) )
    {
       // cout << endl << DT_cov_eig(0) << "\t" << DT_cov_eig(5) << "\t" << err << endl;
        return false;
    }


    return true;
}

//位姿优化函数，高斯牛顿迭代法
void StereoFrameHandler::optimizePose()
{

    // definitions
    Matrix4d DT, DT_;//DT_是没有移除外点时求得的位姿相对量
    Matrix6d DT_cov; //协方差矩阵就是海塞矩阵的逆
    double   err = numeric_limits<double>::max(), e_prev;
    err = -1.0;
    
    // set init pose (depending on the use of prior information or not, and on the goodness of previous solution)
    //用恒速运动模型估算位姿 （并不使用）
    // optimize 因为默认不使用恒速模型，这里初始值DT和DT_都是单位矩阵 
    if( Config::useMotionModel() )
    {
        //prev_frame->DT;一直是单位矩阵啊
        DT     = prev_frame->DT;
        DT_cov = prev_frame->DT_cov;
        e_prev = prev_frame->err_norm;
        //判断先验信息的好坏
        if( !isGoodSolution(DT,DT_cov,e_prev) )
            DT = Matrix4d::Identity();
    }
    else
    {
        DT = Matrix4d::Identity();
    }

    // optimization mode
    int mode = 0;   // GN - GNR - LM

    // solver
    if( n_inliers >= Config::minFeatures() )
    {
        // 这里DT_的作用是为下面剔除外点提供一个位姿
        DT_ = DT;
        //P_ = Pc_current = DT * Pc_previous
        if( mode == 0 )      gaussNewtonOptimization(DT_,DT_cov,err,Config::maxIters());
        else if( mode == 1 ) gaussNewtonOptimizationRobust(DT_,DT_cov,err,Config::maxIters());
        else if( mode == 2 ) levenbergMarquardtOptimization(DT_,DT_cov,err,Config::maxIters());
        // remove outliers (implement some logic based on the covariance's eigenvalues and optim error)
        if( isGoodSolution(DT_,DT_cov,err) )
        {
            //cout<<"  P_L: "<<curr_frame->points_l.size()<<"  P_R: "<<curr_frame->points_r.size()<<"  S_P: "<<curr_frame->stereo_pt.size()<<"  M_P"<<matched_pt.size()<<endl;
            //cout<<"  L_L: "<<curr_frame->lines_l.size()<<"  L_R: "<<curr_frame->lines_r.size()<<"  S_L: "<<curr_frame->stereo_ls.size()<<"  M_L"<<matched_ls.size()<<endl;
            removeOutliers(DT_);
            //gfPointSeclet(DT_);
            //gfLineSeclet();
            // refine without outliers
            // 去除外点之后，再进行一次优化
            if( n_inliers >= Config::minFeatures() )
            {
                //这里的DT初始值还是单位矩阵（不使用恒速模型）
                if( mode == 0 )      gaussNewtonOptimization(DT,DT_cov,err,Config::maxItersRef());
                else if( mode == 1 ) gaussNewtonOptimizationRobust(DT,DT_cov,err,Config::maxItersRef());
                else if( mode == 2 ) levenbergMarquardtOptimization(DT,DT_cov,err,Config::maxItersRef());
            }
            else
            {
                DT     = Matrix4d::Identity();
                cout << "[StVO] not enough inliers (after removal)" << endl;
            }
        }
        else
        {
            gaussNewtonOptimizationRobust(DT,DT_cov,err,Config::maxItersRef());
            //DT     = Matrix4d::Identity();
            //cout << "[StVO] optimization didn't converge" << endl;
        }
    }
    else
    {
        DT     = Matrix4d::Identity();
        cout << "[StVO] not enough inliers (before optimization)" << endl;
    }


    // set estimated pose
    if( isGoodSolution(DT,DT_cov,err) && DT != Matrix4d::Identity() )
    {
        //expmap_se3(logmap_se3这里好像不是很有意义
        //curr_frame->DT=Tpre_cur
        curr_frame->DT       = expmap_se3(logmap_se3( inverse_se3(DT) ));
        //cout<<"Display: curr_frame->DT\n"<<expmap_se3(logmap_se3( inverse_se3(DT) ))<<"\n\n"<<inverse_se3(DT)<<endl;
        //Tpre_cur的协方差（在迭代的时候用信息矩阵的逆代替）
        curr_frame->DT_cov   = DT_cov;
        curr_frame->err_norm = err;
        //更新 prev_frame->T_kf_f在updateFrame里更新，但每当关键帧插入时都会变成单位矩阵。并且在updateFrame里，prev_frame会被整体更新
        curr_frame->T_kf_f      = expmap_se3(logmap_se3( prev_frame->T_kf_f * curr_frame->DT ));
        //cout<<"Display: curr_frame->T_kf_f\n"<<expmap_se3(logmap_se3( prev_frame->T_kf_f * curr_frame->DT ))<<"\n\n"<<prev_frame->T_kf_f * curr_frame->DT<<endl;
        //curr_frame->Tkf_f_cov=prev_frame->Tkf_f_cov+ad(prev_frame->T_kf_f)*DT_cov*ad(prev_frame->T_kf_f)^T
        curr_frame->Tkf_f_cov  = unccomp_se3( prev_frame->T_kf_f, prev_frame->Tkf_f_cov, DT_cov );
        SelfAdjointEigenSolver<Matrix6d> eigensolver(DT_cov);
        curr_frame->DT_cov_eig = eigensolver.eigenvalues();
        
//         cout<<"qiao"<<endl;
//         logT(curr_frame->DT);
//         logT(curr_frame->T_kf_f);

    }
    else
    {
        //setAsOutliers();
        curr_frame->DT       = Matrix4d::Identity();
        curr_frame->DT_cov   = Matrix6d::Zero();
        curr_frame->err_norm = -1.0;
        curr_frame->T_kf_f      = prev_frame->T_kf_f;
        curr_frame->Tkf_f_cov  = prev_frame->Tkf_f_cov;
        curr_frame->DT_cov_eig = Vector6d::Zero();
    }
}

void StereoFrameHandler::resetOutliers() {

    for (auto pt : matched_pt)
        pt->inlier = true;
    for (auto ls : matched_ls)
        ls->inlier = true;

    n_inliers_pt = matched_pt.size();
    n_inliers_ls = matched_ls.size();
    n_inliers    = n_inliers_pt + n_inliers_ls;
}

void StereoFrameHandler::setAsOutliers() {

    for (auto pt : matched_pt)
        pt->inlier = false;
    for (auto ls : matched_ls)
        ls->inlier = false;

    n_inliers_pt = 0;
    n_inliers_ls = 0;
    n_inliers    = 0;
}

//用于根据已知的转换矩阵，去除特征点和特征线中的外点
void StereoFrameHandler::removeOutliers(Matrix4d DT)
{

    //TODO: if not usig mad stdv, use just a fixed threshold (sqrt(7.815)) to filter outliers (with a single for loop...)

    if (Config::hasPoints()) {
        // point features
        vector<double> res_p;
        //按照非线性优化的结果重新把每个特征点的投影误差计算一下
        res_p.reserve(matched_pt.size());
        int iter = 0;
        for( auto it = matched_pt.begin(); it!=matched_pt.end(); it++, iter++)
        {
            // projection error
            Vector3d P_ = DT.block(0,0,3,3) * (*it)->P + DT.col(3).head(3);     
            Vector2d pl_proj = cam->projection( P_ );                          
            res_p.push_back( ( pl_proj - (*it)->pl_obs ).norm() * sqrt((*it)->sigma2) );       
            //res_p.push_back( ( pl_proj - (*it)->pl_obs ).norm() );
        }
        // estimate robust parameters
        double p_stdv, p_mean, inlier_th_p;
        vector_mean_stdv_mad( res_p, p_mean, p_stdv );
        //default 1.0
        inlier_th_p = Config::inlierK() * p_stdv;
        //inlier_th_p = sqrt(7.815);
        //cout << endl << p_mean << " " << p_stdv << "\t" << inlier_th_p << endl;
        // filter outliers
        // 去除点特征集的外点
        iter = 0;
        //一σ原则
        for( auto it = matched_pt.begin(); it!=matched_pt.end(); it++, iter++)
        {
            if( (*it)->inlier && fabs(res_p[iter]-p_mean) > inlier_th_p )
            {
                (*it)->inlier = false;
                n_inliers--;
                n_inliers_pt--;
            }
        }
    }

    if (Config::hasLines()) {
        // line segment features
        vector<double> res_l;
        res_l.reserve(matched_ls.size());
        int iter = 0;
        for( auto it = matched_ls.begin(); it!=matched_ls.end(); it++, iter++)
        {
            // projection error
            Vector3d sP_ = DT.block(0,0,3,3) * (*it)->sP + DT.col(3).head(3);   //线特征的一个端点sP
            Vector3d eP_ = DT.block(0,0,3,3) * (*it)->eP + DT.col(3).head(3);   //线特征的另一个端点eP
            Vector2d spl_proj = cam->projection( sP_ );                         //3D点sP在图像平面上的投影spl_proj
            Vector2d epl_proj = cam->projection( eP_ );                         //3D点sP在图像平面上的投影spl_proj   
            Vector3d l_obs    = (*it)->le_obs;
            Vector2d err_li;
            err_li(0) = l_obs(0) * spl_proj(0) + l_obs(1) * spl_proj(1) + l_obs(2);     //投影点spl_proj到投影线的距离
            err_li(1) = l_obs(0) * epl_proj(0) + l_obs(1) * epl_proj(1) + l_obs(2);     //投影点epl_proj到投影线的距离
            res_l.push_back( err_li.norm() * sqrt( (*it)->sigma2 ) );
            //res_l.push_back( err_li.norm() );
        }

        // estimate robust parameters
        double l_stdv, l_mean, inlier_th_l;
        vector_mean_stdv_mad( res_l, l_mean, l_stdv );
        inlier_th_l = Config::inlierK() * l_stdv;
        //inlier_th_p = sqrt(7.815);
        //cout << endl << l_mean << " " << l_stdv << "\t" << inlier_th_l << endl << endl;

        // filter outliers
        // 去除线特征集的外点
        iter = 0;
        for( auto it = matched_ls.begin(); it!=matched_ls.end(); it++, iter++)
        {
            if( fabs(res_l[iter]-l_mean) > inlier_th_l  && (*it)->inlier )
            {
                (*it)->inlier = false;
                n_inliers--;
                n_inliers_ls--;
            }
        }
    }
    
    //    cout<<"qiao"<<n_inliers_pt<<" line "<<n_inliers_ls<<endl;

    if (n_inliers != (n_inliers_pt + n_inliers_ls))
        throw runtime_error("[StVO; stereoFrameHandler] Assetion failed: n_inliers != (n_inliers_pt + n_inliers_ls)");
}
// 
void StereoFrameHandler::gfPointSeclet(Matrix4d DT)
{
    Timer timer;
    int size=n_inliers_pt;
    int discardNUm=n_inliers_pt/2;
    Matrix6d HcTHc=Matrix6d::Zero();
    Matrix6d HcTHc_=Matrix6d::Zero();
    timer.start();
    if(n_inliers_pt<=100) return;
    if (Config::hasPoints()) {
        for( auto it = matched_pt.begin(); it!=matched_pt.end(); it++)
        {
            (*it)->Hx=getPointHx(logmap_se3(DT),(*it)->P_obs);
            (*it)->Hp=getPointHp(logmap_se3(DT),(*it)->P_obs);
            (*it)->Hc=pInv((*it)->Hp)*((*it)->Hx);  
            (*it)->HcTHc=(*it)->Hc.transpose()*(*it)->Hc;  
        }
        
        for( auto it = matched_pt.begin(); it!=matched_pt.end(); it++)
            if( (*it)->inlier )
                HcTHc= HcTHc+(*it)->HcTHc;
            
        while(1)
        {
            double value_min=std::numeric_limits<double>::max();
            PointFeature* discardId;
            for( auto it = matched_pt.begin(); it!=matched_pt.end(); it++)
            {
                if( (*it)->inlier )
                {
                    HcTHc_=HcTHc-(*it)->HcTHc;
                    double det=HcTHc_.determinant();
                    //找到去掉该点后，HcTHc_的行列式值最小的那个
                    if(det<value_min)
                    {
                        value_min=det;
                        discardId=*it;
                    }
                }
            }
            if(!discardId)
            {
                //更新总集
                HcTHc=HcTHc-discardId->HcTHc;
                //去除内点
                discardId->inlier = false;
            }
            n_inliers--;
            n_inliers_pt--;
            discardNUm--;
            if(discardNUm==0) break;
        }
    }
    cout<<"time: "<<timer.stop()<<endl;
}

void StereoFrameHandler::gfLineSeclet()
{
    Timer timer;
    int size=n_inliers_ls;
    int discardNUm=n_inliers_ls/2;
    Matrix6d info_pose=Matrix6d::Zero();
    Matrix6d info_pose_=Matrix6d::Zero();
    timer.start();
    if(n_inliers_ls<=35) return;
    
    if (Config::hasLines()) {
        for( auto it = matched_ls.begin(); it!=matched_ls.end(); it++)
            getPoseInfoOnLine(*it);
        
        for( auto it = matched_ls.begin(); it!=matched_ls.end(); it++)
            if( (*it)->inlier )
                info_pose= info_pose+(*it)->info_pose;
            
        while(1)
        {
            double value_min=std::numeric_limits<double>::max();
            LineFeature* discardId;
            for( auto it = matched_ls.begin(); it!=matched_ls.end(); it++)
            {
                if( (*it)->inlier )
                {
                    info_pose_=info_pose-(*it)->info_pose;
                    double det=info_pose_.determinant();
                    //找到去掉该点后，info_pose_的行列式值最小的那个
                    if(det<value_min)
                    {
                        value_min=det;
                        discardId=*it;
                    }
                }
            }
            //更新总集
            if(!discardId)
            {
                info_pose=info_pose-discardId->info_pose;
                //去除内点
                discardId->inlier = false;
            }
            
            n_inliers--;
            n_inliers_ls--;
            discardNUm--;
            
            if(discardNUm==0||n_inliers_ls==0) break;
            if(discardNUm<0||n_inliers_ls<0)
            {
                
                cout<<"error"<<discardNUm<<'\t'<<n_inliers_ls<<endl;
            }
        }
    }
    cout<<"time: "<<timer.stop()<<endl;
}

void StereoFrameHandler::prefilterOutliers(Matrix4d DT)
{

    vector<double> res_p, res_l, ove_l, rob_res_p, rob_res_l;

    // point features
    int iter = 0;
    for( auto it = matched_pt.begin(); it!=matched_pt.end(); it++, iter++)
    {
        // projection error
        Vector3d P_ = DT.block(0,0,3,3) * (*it)->P + DT.col(3).head(3);
        Vector2d pl_proj = cam->projection( P_ );
        res_p.push_back( ( pl_proj - (*it)->pl_obs ).norm() * sqrt((*it)->sigma2) );
    }

    // line segment features
    iter = 0;
    for( auto it = matched_ls.begin(); it!=matched_ls.end(); it++, iter++)
    {
        // projection error
        Vector3d sP_ = DT.block(0,0,3,3) * (*it)->sP + DT.col(3).head(3);
        Vector3d eP_ = DT.block(0,0,3,3) * (*it)->eP + DT.col(3).head(3);
        Vector2d spl_proj = cam->projection( sP_ );
        Vector2d epl_proj = cam->projection( eP_ );
        Vector3d l_obs    = (*it)->le_obs;
        Vector2d err_li;
        err_li(0) = l_obs(0) * spl_proj(0) + l_obs(1) * spl_proj(1) + l_obs(2);
        err_li(1) = l_obs(0) * epl_proj(0) + l_obs(1) * epl_proj(1) + l_obs(2);
        res_l.push_back( err_li.norm() * sqrt( (*it)->sigma2 ) );
    }

    // estimate mad standard deviation
    double p_mad = vector_stdv_mad( res_p );
    double l_mad = vector_stdv_mad( res_l );
    double p_mean = vector_mean_mad( res_p, p_mad, Config::inlierK() );
    double l_mean = vector_mean_mad( res_l, l_mad, Config::inlierK() );
    double inlier_th_p =  Config::inlierK() * p_mad;
    double inlier_th_l =  Config::inlierK() * l_mad;
    p_mean = 0.0;
    l_mean = 0.0;

    // filter outliers
    iter = 0;
    for( auto it = matched_pt.begin(); it!=matched_pt.end(); it++, iter++)
    {
        if( res_p[iter] > inlier_th_p && (*it)->inlier )
        {
            (*it)->inlier = false;
            n_inliers--;
            n_inliers_pt--;
        }
    }
    iter = 0;
    for( auto it = matched_ls.begin(); it!=matched_ls.end(); it++, iter++)
    {
        if( res_l[iter] > inlier_th_l  && (*it)->inlier )
        {
            (*it)->inlier = false;
            n_inliers--;
            n_inliers_ls--;
        }
    }

}

/*  slam functions  */
//判断是否需要一个新的关键帧
bool StereoFrameHandler::needNewKF()
{

    // if the previous KF was a KF, update the entropy_first_prevKF value
    // 如果上一帧是关键帧，那么更新熵值 计算h(i,i+1)的熵值，参见论文
    if( prev_f_iskf )
    {
        if( curr_frame->DT_cov.determinant() != 0.0 )
        {
            entropy_first_prevKF = 3.0*(1.0+log(2.0*acos(-1))) + 0.5*log( curr_frame->DT_cov.determinant() );
            prev_f_iskf = false;
        }
        else
        {
            entropy_first_prevKF = -999999999.99;
            prev_f_iskf = false;
        }
    }

    // check geometric distances from previous KF
    //DT是T_prevKF_curr 和类里的DT不同
    //这里的T_prevKF一直是单位矩阵,这里应该是有问题的？
    Matrix4d DT = inverse_se3( curr_frame->T_kf_f ) * T_prevKF;
    Vector6d dX = logmap_se3( DT );

    double t = dX.head(3).norm();
    //转换成角度
    double r = dX.tail(3).norm() * 180.f / CV_PI;

    // check cumulated covariance from previous KF
    //求上一个关键帧的伴随
    Matrix6d adjTprevkf = adjoint_se3( T_prevKF );
    //？？？不是很懂，用DT的逆的伴随去更新协方差
    Matrix6d covDTinv   = uncTinv_se3( curr_frame->DT, curr_frame->DT_cov );
    //这个量原来是累加得到的，一直累加到满足条件为止，别处有复位（currFrameIsKF()） 
    cov_prevKF_currF += adjTprevkf * covDTinv * adjTprevkf.transpose();
    //将代表不确定性的协方差转化为一个标量，称之为entropy
    double entropy_curr  = 3.0*(1.0+log(2.0*acos(-1))) + 0.5*log( cov_prevKF_currF.determinant() );
    //计算当前帧的entropy和前面第一个关键帧的entropy的比值entropy_ratio
    double entropy_ratio = entropy_curr / entropy_first_prevKF;

    //cout << endl << curr_frame->DT     << endl << endl;
    //cout << endl << curr_frame->DT_cov << endl << endl;
    //cout << endl << cov_prevKF_currF   << endl << endl;

    // decide if a new KF is needed
    //如果比值entropy_ratio是在实数范围内或者小于Config::minEntropyRatio()，就把当前帧作为一个新的关键帧插入到系统中：
    if( entropy_ratio < Config::minEntropyRatio() || std::isnan(entropy_ratio) || std::isinf(entropy_ratio) ||
        ( curr_frame->DT_cov == Matrix6d::Zero() && curr_frame->DT == Matrix4d::Identity() ) ||
        t > Config::maxKFTDist() || r > Config::maxKFRDist() || N_prevKF_currF > 10 )
    {
//         cout << endl << "Entropy ratio: " << entropy_ratio   << "\t" << t << " " << r << " " << N_prevKF_currF << endl;
        return true;
    }
    else
    {
//         cout << endl << "No new KF needed: " << entropy_ratio << "\t" << entropy_curr << " " << entropy_first_prevKF
//              << " " << cov_prevKF_currF.determinant() << "\t" << t << " " << r << " " << N_prevKF_currF << endl << endl;
        N_prevKF_currF++;
        return false;
    }
}

// update KF in StVO
void StereoFrameHandler::currFrameIsKF()
{

    // restart point indices
    int idx_pt = 0;
    for( auto it = curr_frame->stereo_pt.begin(); it != curr_frame->stereo_pt.end(); it++)
    {
        (*it)->idx = idx_pt;
        idx_pt++;
    }

    // restart line indices
    int idx_ls = 0;
    for( auto it = curr_frame->stereo_ls.begin(); it != curr_frame->stereo_ls.end(); it++)
    {
        (*it)->idx = idx_ls;
        idx_ls++;
    }

    // update KF
    //// 把当前帧的位姿和协方差矩阵都设置为单位矩阵
    curr_frame->T_kf_f     = Matrix4d::Identity();
    curr_frame->Tkf_f_cov = Matrix6d::Identity();

    // update SLAM variables for KF decision
    // 更新前一个关键帧的位姿，以及前一个关键帧到当前关键帧的协方差矩阵
    T_prevKF = curr_frame->T_kf_f; //这里岂不是一直是单位矩阵了？？？有没有问题
    cov_prevKF_currF = Matrix6d::Zero();
    prev_f_iskf = true;
    N_prevKF_currF = 0;

}

/* Debug functions */

void StereoFrameHandler::plotLeftPair() {

    cout << "Prev frame: " << prev_frame->stereo_pt.size() << endl;
    cout << "Curr frame: " << curr_frame->stereo_pt.size() << endl;
    cout << "Matched: " << matched_pt.size() << endl;
    cout << "Inliers: " << n_inliers_pt << endl;

    Mat im1, im2;

    if (prev_frame->img_l.channels() == 1) {
        im1 = prev_frame->img_l;
    } else if (prev_frame->img_l.channels() == 3 || prev_frame->img_l.channels() == 4)
        cvtColor(prev_frame->img_l, im1, COLOR_BGR2GRAY);
    else
        throw runtime_error("[plotLeftPair] Unsupported number of (prev) image channels");
    im1.convertTo(im1, CV_8UC1);
    cvtColor(im1, im1, COLOR_GRAY2BGR);

    if (curr_frame->img_l.channels() == 1) {
        im2 = curr_frame->img_l;
    } else if (curr_frame->img_l.channels() == 3 || curr_frame->img_l.channels() == 4)
        cvtColor(curr_frame->img_l, im2, COLOR_BGR2GRAY);
    else
        throw runtime_error("[plotLeftPair] Unsupported number of (curr) image channels");
    im2.convertTo(im2, CV_8UC1);
    cvtColor(im2, im2, COLOR_GRAY2BGR);

    Scalar color;

    //plot stereo features
    color = Scalar(0, 255, 0);
    for (auto pt : prev_frame->stereo_pt)
        cv::circle(im1, cv::Point(pt->pl(0), pt->pl(1)), 3, color, -1);

    color = Scalar(0, 0, 255);
    for (auto pt : curr_frame->stereo_pt)
        cv::circle(im2, cv::Point(pt->pl(0), pt->pl(1)), 3, color, -1);

    //plot matched features
    random_device rnd_dev;
    mt19937 rnd(rnd_dev());
    uniform_int_distribution<int> color_dist(0, 255);

    for (auto pt : matched_pt) {
        color = Scalar(color_dist(rnd), color_dist(rnd), color_dist(rnd));
        cv::circle(im1, cv::Point(pt->pl(0), pt->pl(1)), 3, color, -1);
        cv::circle(im2, cv::Point(pt->pl_obs(0), pt->pl_obs(1)), 3, color, -1);
    }

    //putsidebyside
    Size sz1 = im1.size();
    Size sz2 = im2.size();

    Mat im(sz1.height, sz1.width+sz2.width, CV_8UC3);

    Mat left(im, Rect(0, 0, sz1.width, sz1.height));
    im1.copyTo(left);

    Mat right(im, Rect(sz1.width, 0, sz2.width, sz2.height));
    im2.copyTo(right);

    imshow("LeftPair", im);
}

void StereoFrameHandler::plotStereoFrameProjerr( Matrix4d DT, int iter )
{

    // create new image to modify it
    Mat img_l_aux_p, img_l_aux_l;
    curr_frame->img_l.copyTo( img_l_aux_p );

    if( img_l_aux_p.channels() == 3 )
        cvtColor(img_l_aux_p,img_l_aux_p,CV_BGR2GRAY);

    cvtColor(img_l_aux_p,img_l_aux_p,CV_GRAY2BGR);
    img_l_aux_p.copyTo( img_l_aux_l );

    // Variables
    Point2f         p,q,r,s;
    double          thick = 1.5;
    int             k = 0, radius  = 3;

    // plot point features
    for( auto pt_it = matched_pt.begin(); pt_it != matched_pt.end(); pt_it++)
    {
        if( (*pt_it)->inlier )
        {
            Vector3d P_ = DT.block(0,0,3,3) * (*pt_it)->P + DT.col(3).head(3);
            Vector2d pl_proj = cam->projection( P_ );
            Vector2d pl_obs  = (*pt_it)->pl_obs;

            p = cv::Point( int(pl_proj(0)), int(pl_proj(1)) );
            circle( img_l_aux_p, p, radius, Scalar(255,0,0), thick);

            q = cv::Point( int(pl_obs(0)), int(pl_obs(1)) );
            circle( img_l_aux_p, p, radius, Scalar(0,0,255), thick);

            line( img_l_aux_p, p, q, Scalar(0,255,0), thick);

        }
    }

    // plot line segment features
    for( auto ls_it = matched_ls.begin(); ls_it != matched_ls.end(); ls_it++)
    {
        if( (*ls_it)->inlier )
        {
            Vector3d sP_ = DT.block(0,0,3,3) * (*ls_it)->sP + DT.col(3).head(3);
            Vector2d spl_proj = cam->projection( sP_ );
            Vector3d eP_ = DT.block(0,0,3,3) * (*ls_it)->eP + DT.col(3).head(3);
            Vector2d epl_proj = cam->projection( eP_ );

            Vector2d spl_obs  = (*ls_it)->spl_obs;
            Vector2d epl_obs  = (*ls_it)->epl_obs;

            p = cv::Point( int(spl_proj(0)), int(spl_proj(1)) );
            q = cv::Point( int(epl_proj(0)), int(epl_proj(1)) );

            r = cv::Point( int(spl_obs(0)), int(spl_obs(1)) );
            s = cv::Point( int(epl_obs(0)), int(epl_obs(1)) );

            line( img_l_aux_l, p, q, Scalar(255,0,0), thick);
            line( img_l_aux_l, r, s, Scalar(0,0,255), thick);

            line( img_l_aux_l, p, r, Scalar(0,255,0), thick);
            line( img_l_aux_l, q, s, Scalar(0,255,0), thick);

            double overlap = prev_frame->lineSegmentOverlap( spl_obs, epl_obs, spl_proj, epl_proj );
            Vector2d mpl_obs = 0.5*(spl_obs+epl_obs);
            mpl_obs(0) += 4;
            mpl_obs(1) += 4;
            putText(img_l_aux_l,to_string(overlap),cv::Point(int(mpl_obs(0)),int(mpl_obs(1))),
                    FONT_HERSHEY_PLAIN, 0.5, Scalar(0,255,255) );


        }
    }

    //string pwin_name = "Iter: " + to_string(iter);
    string pwin_name = "Points";
    imshow(pwin_name,img_l_aux_p);
    string lwin_name = "Lines";
    imshow(lwin_name,img_l_aux_l);
    waitKey(0);


}

void StereoFrameHandler::optimizePoseDebug()
{

    // definitions
    Matrix6d DT_cov;
    Matrix4d DT, DT_;
    Vector6d DT_cov_eig;
    double   err;

    // set init pose
    DT     = prev_frame->DT;
    
    DT_cov = prev_frame->DT_cov;

    DT = Matrix4d::Identity();
    DT_cov = Matrix6d::Zero();

    // solver
    if( n_inliers > Config::minFeatures() )
    {
        // optimize
        DT_ = DT;
        gaussNewtonOptimizationRobustDebug(DT_,DT_cov,err,Config::maxIters());
        // remove outliers (implement some logic based on the covariance's eigenvalues and optim error)
        if( is_finite(DT_) && err != -1.0 )
        {
            removeOutliers(DT_);
            // refine without outliers
            if( n_inliers > Config::minFeatures() )
                gaussNewtonOptimizationRobustDebug(DT,DT_cov,err,Config::maxItersRef());
            else
            {
                DT     = Matrix4d::Identity();
                DT_cov = Matrix6d::Zero();
            }

        }
        else
        {
            DT     = Matrix4d::Identity();
            DT_cov = Matrix6d::Zero();
        }
    }
    else
    {
        DT     = Matrix4d::Identity();
        DT_cov = Matrix6d::Zero();
    }

    // set estimated pose
    if( is_finite(DT) && err != -1.0 )
    {
        curr_frame->DT     = inverse_se3( DT );
        curr_frame->T_kf_f    = prev_frame->T_kf_f * curr_frame->DT;
        curr_frame->DT_cov = DT_cov;
        SelfAdjointEigenSolver<Matrix6d> eigensolver(DT_cov);
        curr_frame->DT_cov_eig = eigensolver.eigenvalues();
        curr_frame->Tkf_f_cov = unccomp_se3( prev_frame->T_kf_f, prev_frame->Tkf_f_cov, DT_cov );
        curr_frame->err_norm   = err;
    }
    else
    {
        curr_frame->DT     = Matrix4d::Identity();
        curr_frame->DT_cov = Matrix6d::Identity();
        curr_frame->err_norm   = -1.0;
        curr_frame->T_kf_f    = prev_frame->T_kf_f;
        curr_frame->Tkf_f_cov= prev_frame->Tkf_f_cov;
        SelfAdjointEigenSolver<Matrix6d> eigensolver(DT_cov);
        curr_frame->DT_cov_eig = eigensolver.eigenvalues();
    }

    // ************************************************************************************************ //
    // 1. IDENTIFY ANOTHER FUCKING CRITERIA FOR DETECTING JUMPS                                         //
    // 2. INTRODUCE VARIABLE TO INDICATE THAT IN THAT KF ESTIMATION YOU'VE JUMPED SEVERAL FRAMES -> PGO //
    // ************************************************************************************************ //
    // cout << endl << prev_frame->DT_cov_eig.transpose() << endl;                                      //
    // cout << endl << prev_frame->DT                     << endl;                                      //
    /*SelfAdjointEigenSolver<Matrix6d> eigensolver(DT_cov);
    Vector6d max_eig = eigensolver.eigenvalues();
    SelfAdjointEigenSolver<Matrix3d> eigensolverT( DT_cov.block(0,0,3,3) );
    SelfAdjointEigenSolver<Matrix3d> eigensolverR( DT_cov.block(3,3,3,3) );
    Vector3d max_eig_t = eigensolverT.eigenvalues();
    Vector3d max_eig_r = eigensolverR.eigenvalues();
    Vector6d shur = logmap_se3(DT);
    cout.setf(ios::fixed,ios::floatfield); cout.precision(8);
    cout << endl << max_eig << endl;
    //cout << endl << max_eig(3) / max_eig(0) << endl;
    //cout << endl << max_eig(4) / max_eig(1) << endl;
    //cout << endl << max_eig(5) / max_eig(2) << endl << endl;
    cout << endl << shur.head(3).norm() << endl;
    cout << endl << shur.tail(3).norm() << endl;
    cout << endl << max_eig_t * shur.head(3).norm()  << endl;
    cout << endl << max_eig_r * shur.tail(3).norm() << endl << endl;
    cout.setf(ios::fixed,ios::floatfield); cout.precision(3);*/
    //getchar();
    // ************************************************************************************************ //

    // set estimated pose
    if( is_finite(DT) )
    {
        curr_frame->DT     = inverse_se3( DT );
        curr_frame->T_kf_f    = prev_frame->T_kf_f * curr_frame->DT;
        curr_frame->DT_cov = DT_cov;
        SelfAdjointEigenSolver<Matrix6d> eigensolver(DT_cov);
        curr_frame->DT_cov_eig = eigensolver.eigenvalues();
        curr_frame->Tkf_f_cov = unccomp_se3( prev_frame->T_kf_f, prev_frame->Tkf_f_cov, DT_cov );
        curr_frame->err_norm   = err;
    }
    else
    {
        curr_frame->DT     = Matrix4d::Identity();
        curr_frame->T_kf_f    = prev_frame->T_kf_f;
        curr_frame->Tkf_f_cov= prev_frame->Tkf_f_cov;
        curr_frame->DT_cov = DT_cov;
        SelfAdjointEigenSolver<Matrix6d> eigensolver(DT_cov);
        curr_frame->DT_cov_eig = eigensolver.eigenvalues();
        curr_frame->err_norm   = -1.0;
    }

}

void StereoFrameHandler::gaussNewtonOptimizationRobustDebug(Matrix4d &DT, Matrix6d &DT_cov, double &err_, int max_iters)
{
    Matrix6d H;
    Vector6d g, DT_inc;
    double err, err_prev = 999999999.9;
    bool solution_is_good = true;
    Matrix4d DT_;

    DT_ = DT;

    // plot initial solution
    plotStereoFrameProjerr(DT,0);
    int iters;
    for( iters = 0; iters < max_iters; iters++)
    {
        // estimate hessian and gradient (select)
        optimizeFunctionsRobust( DT, H, g, err );
        // if the difference is very small stop
        if( ( fabs(err-err_prev) < Config::minErrorChange() ) || ( err < Config::minError()) )// || err > err_prev )
            break;
        // update step
        //LDLT<Matrix6d> solver(H);
        ColPivHouseholderQR<Matrix6d> solver(H);
        DT_inc = solver.solve(g);
        if( solver.logAbsDeterminant() < 10.0 || solver.info() != Success )
        {
            solution_is_good = false;
            cout << endl << "Cuidao shur" << endl;
            getchar();
            break;
        }
        DT  << DT * inverse_se3( expmap_se3(DT_inc) );
        // plot each iteration
        plotStereoFrameProjerr(DT,iters+1);
        // if the parameter change is small stop (TODO: change with two parameters, one for R and another one for t)
        if( DT_inc.norm() < Config::minErrorChange() )
            break;
        // update previous values
        err_prev = err;
    }

    if( solution_is_good )
    {
        DT_cov = H.inverse();  //DT_cov = Matrix6d::Identity();
        err_   = err;
    }
    else
    {
        DT = DT_;
        err_ = -1.0;
        DT_cov = Matrix6d::Identity();
    }

}

Matrix26d StereoFrameHandler:: getPointHx(Vector6d x,Vector3d Pc)
{
    Matrix23d dePc;//投影函数对PC的雅克比
    Matrix36d deDT;//Pc对李代数的扰动的雅克比
//     Matrix6d  Jl;  //SE的左雅克比矩阵
//     Matrix3d  jl;  //SO上的左雅克比矩阵
//     Matrix3d  Ql;  //参见Jl的求法
    double fx=cam->getFx();
    double fy=cam->getFy();
    double cx=cam->getCx();
    double cy=cam->getCy();
    
    double pc_x=Pc(0);
    double pc_y=Pc(1);
    double pc_z=Pc(2);
    double pc_z2=pc_z*pc_z;
    
    dePc<<fx/pc_z,0,-fx*pc_x/pc_z2,0,fy/pc_z,-fy*pc_y/pc_z2;
    deDT.block(0,0,3,3)=Matrix3d::Identity();
    deDT.block(0,3,3,3)=-skew(Pc);
    
    return dePc*deDT;
}


Matrix23d StereoFrameHandler:: getPointHp(Vector6d x,Vector3d Pc)
{
    Matrix23d dePc;//投影函数对PC的雅克比
    Matrix3d dePw;//Pc对李代数的扰动的雅克比
//     Matrix6d  Jl;  //SE的左雅克比矩阵
//     Matrix3d  jl;  //SO上的左雅克比矩阵
//     Matrix3d  Ql;  //参见Jl的求法
    double fx=cam->getFx();
    double fy=cam->getFy();
    double cx=cam->getCx();
    double cy=cam->getCy();
    
    double pc_x=Pc(0);
    double pc_y=Pc(1);
    double pc_z=Pc(2);
    double pc_z2=pc_z*pc_z;
    
    dePc<<fx/pc_z,0,-fx*pc_x/pc_z2,0,fy/pc_z,-fy*pc_y/pc_z2;
    //旋转向量
    dePw=fast_skewexp(x.tail(3));
 
    
    return dePc*dePw;
}

void StereoFrameHandler::getPoseInfoOnLine(LineFeature * line) {
    Matrix3d J_proj;
    Matrix4d DT_inv = prev_frame->DT.inverse();
    Vector2d J_loss(line->le(0),line->le(1));
    Matrix3d J_dt = DT_inv.block(0,0,3,3);
    Vector3d cur_sP_tmp = J_dt * line->sP + DT_inv.col(3).head(3);//预测上一帧的两个端点在当前帧相机坐标系的位置
    Vector3d cur_eP_tmp = J_dt * line->eP + DT_inv.col(3).head(3);
    
	// u v d（视差）对 X Y Z的雅克比矩阵 实际只用到了u v对XYZ的雅克比矩阵
    getStereoJacob2D_3D(cur_sP_tmp[0], cur_sP_tmp[1], cur_sP_tmp[2],cam->getFx(),cam->getB(),J_proj);
    double cov_sR_tmp = J_loss.transpose() *
            ( J_proj.block(0,0,2,3) * J_dt * line->covSpt3D
              * J_dt.transpose() * J_proj.block(0,0,2,3).transpose() )
            * J_loss;
    
    getStereoJacob2D_3D(cur_eP_tmp[0], cur_eP_tmp[1], cur_eP_tmp[2],cam->getFx(),cam->getB(),J_proj);
    double cov_eR_tmp = J_loss.transpose() *
            ( J_proj.block(0,0,2,3) * J_dt * line->covEpt3D
              * J_dt.transpose() * J_proj.block(0,0,2,3).transpose() )
            * J_loss;
    
    Matrix<double, 2, 2> cov_r_temp;
    cov_r_temp << cov_sR_tmp, 0, 0, cov_eR_tmp;
	Vector2d err_i;
    Vector6d J_aux;
    getLineJacobi(line,DT_inv,err_i,J_aux);
    Vector2d Je(err_i(0)/err_i.norm(),err_i(1)/err_i.norm());
    Matrix<double, 1, 1> cov_r = Je.transpose()*cov_r_temp*Je;
	//信息矩阵不满秩 cov_r.inverse()不可以近似为单位矩阵
    line->info_pose = J_aux * cov_r.inverse() * J_aux.transpose();
}


}

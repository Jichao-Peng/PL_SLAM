#include <stereoFrameHandler.h>
#include <random>
#include "matching.h"
#include "timer.h"
#ifdef ERROR_FORM_3
namespace StVO{
    //这个没有使用g2o的优化库，就是根据GN优化过程自己写的一段优化代码
void StereoFrameHandler::gaussNewtonOptimization(Matrix4d &DT, Matrix6d &DT_cov, double &err_, int max_iters)
{
    Matrix6d H;
    Vector6d g, DT_inc,DT_inc2;//DT_inc是Hx=g的解
    double err, err_prev = 999999999.9;
    static float iterTimes=0;
    static float time=0;
    int iterTimeTemp=0;
    time++;
    int iters;
    for( iters = 0; iters < max_iters; iters++)
    {
        // estimate hessian and gradient (select)
        // 使用所有的匹配特征估计海森矩阵和梯度；用的是扰动模型
        //P_ = Pc_current = DT * Pc_previous
        optimizeFunctions( DT, H, g, err );
        if (err > err_prev) {
            if (iters > 0)
                break;
            err_ = -1.0;
            return;
        }
        // if the difference is very small stop
        if( ( ( err < Config::minError()) || abs(err-err_prev) < Config::minErrorChange() ) ) {
            cout << "[StVO] Small optimization improvement" << endl;
            break;
        }
        // update step
        ColPivHouseholderQR<Matrix6d> solver(H);
        DT_inc = solver.solve(g);
        //DT_inc2=solver.solve(-g);
        // 增量更新  DT Tcur_pre  这里右乘的原因是算H和g的时候，g忘取了一个负号（原作者真坑）
        //DT  << expmap_se3(DT_inc2)*DT;
        DT  << DT * inverse_se3( expmap_se3(DT_inc) );
        //cout<<"left and right multiply"<<DT<<"\n\n"<<expmap_se3(DT_inc2)*DT<<"\n\n"<<DT-expmap_se3(DT_inc2)*DT<<endl;
        // if the parameter change is small stop
        //1e-7        # min. error change to stop the optimization
        if( DT_inc.head(3).norm() < Config::minErrorChange() && DT_inc.tail(3).norm() < Config::minErrorChange()) {
            cout << "[StVO] Small optimization solution variance" << endl;
            break;
        }
        //平均进行四次迭代
        //cout<<err_prev-err<<' ';
        // update previous values
        err_prev = err;
        iterTimeTemp++;
    }
    //计算平均迭代次数
    iterTimes=(iterTimes*(time-1)+iterTimeTemp)/time;
    //cout<<iterTimes<<endl;
    DT_cov = H.inverse();  //DT_cov = Matrix6d::Identity(); 协方差矩阵就是海塞矩阵的逆
    err_   = err;
}


void StereoFrameHandler::gaussNewtonOptimizationRobust(Matrix4d &DT, Matrix6d &DT_cov, double &err_, int max_iters)
{

    Matrix4d DT_;
    Matrix6d H;
    Vector6d g, DT_inc;
    double err, err_prev = 999999999.9;
    bool solution_is_good = true;
    DT_ = DT;

    // plot initial solution
    int iters;
    for( iters = 0; iters < max_iters; iters++)
    {
        // estimate hessian and gradient (select)
        optimizeFunctionsRobust( DT, H, g, err );
        // if the difference is very small stop
        // 如果计算出来的误差变化很小，就停止迭代
        if( ( fabs(err-err_prev) < Config::minErrorChange() ) || ( err < Config::minError()) )// || err > err_prev )
            break;
        // update step
        // 更新步骤，就是求解delta(x)H=g
        ColPivHouseholderQR<Matrix6d> solver(H);
        DT_inc = solver.solve(g);
        if( solver.logAbsDeterminant() < 0.0 || solver.info() != Success )
        {
            solution_is_good = false;
            break;
        }
        DT  << DT * inverse_se3( expmap_se3(DT_inc) );
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

void StereoFrameHandler::levenbergMarquardtOptimization(Matrix4d &DT, Matrix6d &DT_cov, double &err_, int max_iters)
{
    Matrix6d H;
    Vector6d g, DT_inc;
    double err, err_prev = 999999999.9;

    double lambda   = 0.000000001;
    double lambda_k = 4.0;

    // form the first iteration
//    optimizeFunctionsRobust( DT, H, g, err );
    optimizeFunctions( DT, H, g, err );

    // initial guess of lambda
    double Hmax = 0.0;
    for( int i = 0; i < 6; i++)
    {
        if( H(i,i) > Hmax || H(i,i) < -Hmax )
            Hmax = fabs( H(i,i) );
    }
    lambda *= Hmax;

    // solve the first iteration
    for(int i = 0; i < 6; i++)
        H(i,i) += lambda;// * H(i,i) ;
    ColPivHouseholderQR<Matrix6d> solver(H);
    DT_inc = solver.solve(g);
    DT  << DT * inverse_se3( expmap_se3(DT_inc) );
    err_prev = err;

    // start Levenberg-Marquardt minimization
    //plotStereoFrameProjerr(DT,0);
    for( int iters = 1; iters < max_iters; iters++)
    {
        // estimate hessian and gradient (select)
//        optimizeFunctionsRobust( DT, H, g, err );
        optimizeFunctions( DT, H, g, err );
        // if the difference is very small stop
        if( ( fabs(err-err_prev) < Config::minErrorChange() ) || ( err < Config::minError()) )
            break;
        // add lambda to hessian
        for(int i = 0; i < 6; i++)
            H(i,i) += lambda;// * H(i,i) ;
        // update step
        ColPivHouseholderQR<Matrix6d> solver(H);
        DT_inc = solver.solve(g);
        // update lambda
        if( err > err_prev )
            lambda /= lambda_k;
        else
        {
            lambda *= lambda_k;
            DT  << DT * inverse_se3( expmap_se3(DT_inc) );
        }
        // plot each iteration
        //plotStereoFrameProjerr(DT,iters+1);
        // if the parameter change is small stop
        if( DT_inc.head(3).norm() < Config::minErrorChange() && DT_inc.tail(3).norm() < Config::minErrorChange())
            break;
        // update previous values
        err_prev = err;
    }

    DT_cov = H.inverse();  //DT_cov = Matrix6d::Identity();
    err_   = err;
}

void StereoFrameHandler::getLineJacobi(const LineFeature* line,Matrix4d DT,Vector2d& err_i,Vector6d& J_aux)
{
    Vector3d sP_ = DT.block(0,0,3,3) * (line)->sP + DT.col(3).head(3);
    Vector2d spl_proj = cam->projection( sP_ );
    Vector3d eP_ = DT.block(0,0,3,3) * (line)->eP + DT.col(3).head(3);
    Vector2d epl_proj = cam->projection( eP_ );
    Vector3d l_obs = (line)->le_obs;
    //计算点线距离误差
    err_i(0) = l_obs(0) * spl_proj(0) + l_obs(1) * spl_proj(1) + l_obs(2);
    err_i(1) = l_obs(0) * epl_proj(0) + l_obs(1) * epl_proj(1) + l_obs(2);
    double err_i_norm = err_i.norm();
    double ds   = err_i(0);
    double de   = err_i(1);
    double lx   = l_obs(0);
    double ly   = l_obs(1);
    // estimate variables for J, H, and g
    // -- start point
    double gx   = sP_(0);
    double gy   = sP_(1);
    double gz   = sP_(2);
    double gz2  = gz*gz;
    double fgz2 = cam->getFx() / std::max(Config::homogTh(),gz2);
    Vector6d Js_aux;
    Js_aux << + fgz2 * lx * gz,
              + fgz2 * ly * gz,
              - fgz2 * ( gx*lx + gy*ly ),
              - fgz2 * ( gx*gy*lx + gy*gy*ly + gz*gz*ly ),
              + fgz2 * ( gx*gx*lx + gz*gz*lx + gx*gy*ly ),
              + fgz2 * ( gx*gz*ly - gy*gz*lx );
    // -- end point
    gx   = eP_(0);
    gy   = eP_(1);
    gz   = eP_(2);
    gz2  = gz*gz;
    fgz2 = cam->getFx() / std::max(Config::homogTh(),gz2);
    Vector6d Je_aux;
    Je_aux << + fgz2 * lx * gz,
              + fgz2 * ly * gz,
              - fgz2 * ( gx*lx + gy*ly ),
              - fgz2 * ( gx*gy*lx + gy*gy*ly + gz*gz*ly ),
              + fgz2 * ( gx*gx*lx + gz*gz*lx + gx*gy*ly ),
              + fgz2 * ( gx*gz*ly - gy*gz*lx );
    //综合两个雅克比矩阵
    // jacobian
    J_aux = ( Js_aux * ds + Je_aux * de ) / std::max(Config::homogTh(),err_i_norm);
}

//这个函数就主要是用于计算海森矩阵和雅克比矩阵以及残差
void StereoFrameHandler:: optimizeFunctions(Matrix4d DT, Matrix6d &H, Vector6d &g, double &e )
{

    // define hessians, gradients, and residuals
    Matrix6d H_l, H_p;
    Vector6d g_l, g_p;
    double   e_l = 0.0, e_p = 0.0, S_l, S_p;
    H   = Matrix6d::Zero(); H_l = H; H_p = H;
    g   = Vector6d::Zero(); g_l = g; g_p = g;
    e   = 0.0;

    // point features
    int N_p = 0;

    //对上一帧的特征点进行遍历
    for( auto it = matched_pt.begin(); it!=matched_pt.end(); it++)
    {
        if( (*it)->inlier )
        {
            //P_ = Pc_current = DT * Pc_previous
            Vector3d P_ = DT.block(0,0,3,3) * (*it)->P + DT.col(3).head(3);
            //h(Pc_current)
            Vector2d pl_proj = cam->projection( P_ );
            // projection error;
            //h(Pc_current)-z
            Vector2d err_i    = pl_proj - (*it)->pl_obs;
            double err_i_norm = err_i.norm();
            // estimate variables for J, H, and g
            double gx   = P_(0);
            double gy   = P_(1);
            double gz   = P_(2);
            double gz2  = gz*gz;
            double fgz2 = cam->getFx() / std::max(Config::homogTh(),gz2);//fx fy 相等
            double dx   = err_i(0);
            double dy   = err_i(1);
            // jacobian
            // jacobian 这一处，把误差转为一维了也就是err_i_norm，err_i_norm对err_i求导再乘上原来二维的雅克比矩阵就变成了现在的一维的。
            Vector6d J_aux;
            J_aux << + fgz2 * dx * gz,
                     + fgz2 * dy * gz,
                     - fgz2 * ( gx*dx + gy*dy ),
                     - fgz2 * ( gx*gy*dx + gy*gy*dy + gz*gz*dy ),
                     + fgz2 * ( gx*gx*dx + gz*gz*dx + gx*gy*dy ),
                     + fgz2 * ( gx*gz*dy - gy*gz*dx );
            //除以一个公因子，按照上一句注释求一下导就知道了
            J_aux = J_aux / std::max(Config::homogTh(),err_i_norm);
            // define the residue
            double s2 = (*it)->sigma2;//s2=1
            // if employing robust cost function
            //设置权重，残差越大，权重越小
            double w  = 1.0 / ( 1.0 + err_i_norm * err_i_norm * s2 );//robustWeightCauchy

            // if down-weighting far samples
            //double zdist   = P_(2) * 1.0 / ( cam->getB()*cam->getFx());
            //if( false ) w *= 1.0 / ( s2 + zdist * zdist );

            // update hessian, gradient, and error
            //这里J本是列向量
            H_p += J_aux * J_aux.transpose() * w;
            g_p += J_aux * err_i_norm * sqrt(s2) * w;
            e_p += err_i_norm * err_i_norm * s2 * w;
            N_p++;
        }
    }

    // line segment features
    int N_l = 0;
    //matched_ls 存储的是上一帧的线特征
    for( auto it = matched_ls.begin(); it!=matched_ls.end(); it++)
    {
        if( (*it)->inlier )
        {
            //
            Vector3d sP_ = DT.block(0,0,3,3) * (*it)->sP + DT.col(3).head(3);
            Vector2d spl_proj = cam->projection( sP_ );
            Vector3d eP_ = DT.block(0,0,3,3) * (*it)->eP + DT.col(3).head(3);
            Vector2d epl_proj = cam->projection( eP_ );
            Vector3d l_obs = (*it)->le_obs;
            // projection error
            Vector2d err_i;
            //计算点线距离误差
            err_i(0) = l_obs(0) * spl_proj(0) + l_obs(1) * spl_proj(1) + l_obs(2);
            err_i(1) = l_obs(0) * epl_proj(0) + l_obs(1) * epl_proj(1) + l_obs(2);
            double err_i_norm = err_i.norm();
            double ds   = err_i(0);
            double de   = err_i(1);
            double lx   = l_obs(0);
            double ly   = l_obs(1);
            // estimate variables for J, H, and g
            // -- start point
            double gx   = sP_(0);
            double gy   = sP_(1);
            double gz   = sP_(2);
            double gz2  = gz*gz;
            double fgz2 = cam->getFx() / std::max(Config::homogTh(),gz2);
            Vector6d Js_aux;
            Js_aux << + fgz2 * lx * gz,
                      + fgz2 * ly * gz,
                      - fgz2 * ( gx*lx + gy*ly ),
                      - fgz2 * ( gx*gy*lx + gy*gy*ly + gz*gz*ly ),
                      + fgz2 * ( gx*gx*lx + gz*gz*lx + gx*gy*ly ),
                      + fgz2 * ( gx*gz*ly - gy*gz*lx );
            // -- end point
            gx   = eP_(0);
            gy   = eP_(1);
            gz   = eP_(2);
            gz2  = gz*gz;
            fgz2 = cam->getFx() / std::max(Config::homogTh(),gz2);
            Vector6d Je_aux, J_aux;
            Je_aux << + fgz2 * lx * gz,
                      + fgz2 * ly * gz,
                      - fgz2 * ( gx*lx + gy*ly ),
                      - fgz2 * ( gx*gy*lx + gy*gy*ly + gz*gz*ly ),
                      + fgz2 * ( gx*gx*lx + gz*gz*lx + gx*gy*ly ),
                      + fgz2 * ( gx*gz*ly - gy*gz*lx );
            //综合两个雅克比矩阵
            // jacobian
            J_aux = ( Js_aux * ds + Je_aux * de ) / std::max(Config::homogTh(),err_i_norm);

            // define the residue
            double s2 = (*it)->sigma2;
            double r = err_i_norm * sqrt(s2);

            // if employing robust cost function
            double w  = 1.0;
            w = robustWeightCauchy(r);

            // estimating overlap between line segments
            bool has_overlap = true;
            double overlap = prev_frame->lineSegmentOverlap( (*it)->spl, (*it)->epl, spl_proj, epl_proj );
            if( has_overlap )
                w *= overlap;

            // if down-weighting far samples
            /*double zdist = max( sP_(2), eP_(2) ) / ( cam->getB()*cam->getFx());
            if( false )
                w *= 1.0 / ( s2 + zdist * zdist );*/

            // update hessian, gradient, and error
            H_l += J_aux * J_aux.transpose() * w;
            g_l += J_aux * r * w;
            e_l += r * r * w;
            N_l++;
        }
    }

    // sum H, g and err from both points and lines
    H = H_p + H_l;
    g = g_p + g_l;
    e = e_p + e_l;

    // normalize error
    e /= (N_l+N_p);

}

void StereoFrameHandler::optimizeFunctionsRobust(Matrix4d DT, Matrix6d &H, Vector6d &g, double &e )
{

    // define hessians, gradients, and residuals
    Matrix6d H_l, H_p;
    Vector6d g_l, g_p;
    double   e_l = 0.0, e_p = 0.0, S_l, S_p;
    H   = Matrix6d::Zero(); H_l = H; H_p = H;
    g   = Vector6d::Zero(); g_l = g; g_p = g;
    e   = 0.0;

    vector<double> res_p, res_l;

    // point features pre-weight computation
    for( auto it = matched_pt.begin(); it!=matched_pt.end(); it++)
    {
        if( (*it)->inlier )
        {
            Vector3d P_ = DT.block(0,0,3,3) * (*it)->P + DT.col(3).head(3);
            Vector2d pl_proj = cam->projection( P_ );
            // projection error
            Vector2d err_i    = pl_proj - (*it)->pl_obs;
            res_p.push_back( err_i.norm());
        }
    }

    // line segment features pre-weight computation
    for( auto it = matched_ls.begin(); it!=matched_ls.end(); it++)
    {
        if( (*it)->inlier )
        {
            Vector3d sP_ = DT.block(0,0,3,3) * (*it)->sP + DT.col(3).head(3);
            Vector2d spl_proj = cam->projection( sP_ );
            Vector3d eP_ = DT.block(0,0,3,3) * (*it)->eP + DT.col(3).head(3);
            Vector2d epl_proj = cam->projection( eP_ );
            Vector3d l_obs = (*it)->le_obs;
            // projection error
            Vector2d err_i;
            err_i(0) = l_obs(0) * spl_proj(0) + l_obs(1) * spl_proj(1) + l_obs(2);
            err_i(1) = l_obs(0) * epl_proj(0) + l_obs(1) * epl_proj(1) + l_obs(2);
            res_l.push_back( err_i.norm() );
        }

    }

    // estimate scale of the residuals
    double s_p = 1.0, s_l = 1.0;
    double th_min = 0.0001;
    double th_max = sqrt(7.815);

    if( false )
    {

        res_p.insert( res_p.end(), res_l.begin(), res_l.end() );
        s_p = vector_stdv_mad( res_p );

        //cout << s_p << "\t";
        //if( res_p.size() < 4*Config::minFeatures() )
            //s_p = 1.0;
        //cout << s_p << endl;

        if( s_p < th_min )
            s_p = th_min;
        if( s_p > th_max )
            s_p = th_max;

        s_l = s_p;

    }
    else
    {

        s_p = vector_stdv_mad( res_p );
        s_l = vector_stdv_mad( res_l );

        if( s_p < th_min )
            s_p = th_min;
        if( s_p > th_max )
            s_p = th_max;

        if( s_l < th_min )
            s_l = th_min;
        if( s_l > th_max )
            s_l = th_max;

    }

    // point features
    int N_p = 0;
    for( auto it = matched_pt.begin(); it!=matched_pt.end(); it++)
    {
        if( (*it)->inlier )
        {
            Vector3d P_ = DT.block(0,0,3,3) * (*it)->P + DT.col(3).head(3);
            Vector2d pl_proj = cam->projection( P_ );
            // projection error
            Vector2d err_i    = pl_proj - (*it)->pl_obs;
            double err_i_norm = err_i.norm();
            // estimate variables for J, H, and g
            double gx   = P_(0);
            double gy   = P_(1);
            double gz   = P_(2);
            double gz2  = gz*gz;
            double fgz2 = cam->getFx() / std::max(Config::homogTh(),gz2);
            double dx   = err_i(0);
            double dy   = err_i(1);
            // jacobian
            Vector6d J_aux;
            J_aux << + fgz2 * dx * gz,
                     + fgz2 * dy * gz,
                     - fgz2 * ( gx*dx + gy*dy ),
                     - fgz2 * ( gx*gy*dx + gy*gy*dy + gz*gz*dy ),
                     + fgz2 * ( gx*gx*dx + gz*gz*dx + gx*gy*dy ),
                     + fgz2 * ( gx*gz*dy - gy*gz*dx );
            J_aux = J_aux / std::max(Config::homogTh(),err_i_norm);
            // define the residue
            double s2 = (*it)->sigma2;
            double r = err_i_norm ;
            // if employing robust cost function
            double w  = 1.0;
            double x = r / s_p;
            w = robustWeightCauchy(x) ;

            // if using uncertainty weights
            //----------------------------------------------------
            if( false )
            {
                Matrix2d covp;
                Matrix3d covP_ = (*it)->covP_an;
                MatrixXd J_Pp(2,3), J_pr(1,2);
                // uncertainty of the projection
                J_Pp  << gz, 0.f, -gx, 0.f, gz, -gy;
                J_Pp  << J_Pp * DT.block(0,0,3,3);
                covp  << J_Pp * covP_ * J_Pp.transpose();
                covp  << covp / std::max(Config::homogTh(),gz2*gz2);               // Covariance of the 3D projection \hat{p} up to f2*b2*sigma2
                covp  = sqrt(s2) * cam->getB()* cam->getFx() * covp;
                covp(0,0) += s2;
                covp(1,1) += s2;
                // Point Uncertainty constants
                /*bsigmaP   = f * baseline * sigmaP;
                bsigmaP   = bsigmaP * bsigmaP;
                bsigmaP_inv   = 1.f / bsigmaP;
                sigmaP2       = sigmaP * sigmaP;
                sigmaP2_inv   = 1.f / sigmaP2;
                // Line Uncertainty constants
                bsigmaL   = baseline * sigmaL;
                bsigmaL   = bsigmaL * bsigmaL;
                bsigmaL_inv   = 1.f / bsigmaL;*/
                // uncertainty of the residual
                J_pr << dx / r, dy / r;
                double cov_res = (J_pr * covp * J_pr.transpose())(0);
                cov_res = 1.0 / std::max(Config::homogTh(),cov_res);
                double zdist   = P_(2) * 1.0 / ( cam->getB()*cam->getFx());

                //zdist   = 1.0 / std::max(Config::homogTh(),zdist);
                /*cout.setf(ios::fixed,ios::floatfield); cout.precision(8);
                cout << endl << cov_res << " " << 1.0 / cov_res << " " << zdist << " " << 1.0 / zdist << " " << 1.0 / (zdist*40.0) << "\t"
                     << 1.0 / ( 1.0 +  cov_res * cov_res + zdist * zdist ) << " \t"
                     << 1.0 / ( cov_res * cov_res + zdist * zdist )
                     << endl;
                cout.setf(ios::fixed,ios::floatfield); cout.precision(3);*/
                //w *= 1.0 / ( cov_res * cov_res + zdist * zdist );
                w *= 1.0 / ( s2 + zdist * zdist );
            }


            //----------------------------------------------------

            // update hessian, gradient, and error
            H_p += J_aux * J_aux.transpose() * w;
            g_p += J_aux * r * w;
            e_p += r * r * w;
            N_p++;
        }
    }

    // line segment features
    int N_l = 0;
    for( auto it = matched_ls.begin(); it!=matched_ls.end(); it++)
    {
        if( (*it)->inlier )
        {
            Vector3d sP_ = DT.block(0,0,3,3) * (*it)->sP + DT.col(3).head(3);
            Vector2d spl_proj = cam->projection( sP_ );
            Vector3d eP_ = DT.block(0,0,3,3) * (*it)->eP + DT.col(3).head(3);
            Vector2d epl_proj = cam->projection( eP_ );
            Vector3d l_obs = (*it)->le_obs;
            // projection error
            Vector2d err_i;
            err_i(0) = l_obs(0) * spl_proj(0) + l_obs(1) * spl_proj(1) + l_obs(2);
            err_i(1) = l_obs(0) * epl_proj(0) + l_obs(1) * epl_proj(1) + l_obs(2);
            double err_i_norm = err_i.norm();
            // estimate variables for J, H, and g
            // -- start point
            double gx   = sP_(0);
            double gy   = sP_(1);
            double gz   = sP_(2);
            double gz2  = gz*gz;
            double fgz2 = cam->getFx() / std::max(Config::homogTh(),gz2);
            double ds   = err_i(0);
            double de   = err_i(1);
            double lx   = l_obs(0);
            double ly   = l_obs(1);
            Vector6d Js_aux;
            Js_aux << + fgz2 * lx * gz,
                      + fgz2 * ly * gz,
                      - fgz2 * ( gx*lx + gy*ly ),
                      - fgz2 * ( gx*gy*lx + gy*gy*ly + gz*gz*ly ),
                      + fgz2 * ( gx*gx*lx + gz*gz*lx + gx*gy*ly ),
                      + fgz2 * ( gx*gz*ly - gy*gz*lx );
            // -- end point
            gx   = eP_(0);
            gy   = eP_(1);
            gz   = eP_(2);
            gz2  = gz*gz;
            fgz2 = cam->getFx() / std::max(Config::homogTh(),gz2);
            Vector6d Je_aux, J_aux;
            Je_aux << + fgz2 * lx * gz,
                      + fgz2 * ly * gz,
                      - fgz2 * ( gx*lx + gy*ly ),
                      - fgz2 * ( gx*gy*lx + gy*gy*ly + gz*gz*ly ),
                      + fgz2 * ( gx*gx*lx + gz*gz*lx + gx*gy*ly ),
                      + fgz2 * ( gx*gz*ly - gy*gz*lx );
            // jacobian
            J_aux = ( Js_aux * ds + Je_aux * de ) / std::max(Config::homogTh(),err_i_norm);
            // define the residue
            double s2 = (*it)->sigma2;
            double r = err_i_norm;
            // if employing robust cost function
            double w  = 1.0;
            double x = r / s_l;
            w = robustWeightCauchy(x) ;
            // estimating overlap between line segments
            bool has_overlap = true;
            double overlap = prev_frame->lineSegmentOverlap( (*it)->spl, (*it)->epl, spl_proj, epl_proj );
            if( has_overlap )
                w *= overlap;

            //----------------- DEBUG: 27/11/2017 ----------------------
            if( false )
            {
                //double cov3d = (*it)->cov3d;
                //cov3d = 1.0;
                //w *= 1.0 / ( 1.0 +  cov3d * cov3d + zdist * zdist );
                double zdist = max( sP_(2), eP_(2) ) / ( cam->getB()*cam->getFx());
                w *= 1.0 / ( s2 + zdist * zdist );
            }
            //----------------------------------------------------------

            // update hessian, gradient, and error
            H_l += J_aux * J_aux.transpose() * w;
            g_l += J_aux * r * w;
            e_l += r * r * w;
            N_l++;
        }

    }

    // sum H, g and err from both points and lines
    H = H_p + H_l;
    g = g_p + g_l;
    e = e_p + e_l;

    // normalize error
    e /= (N_l+N_p);

}

    
}
#endif



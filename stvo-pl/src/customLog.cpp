#include "customLog.h"


void logT(Matrix4d T)
{
    Eigen::Matrix3d rotation_matrix=T.block(0,0,3,3);
    Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles ( 2,1,0 )/CV_PI*180.0; // ZYX顺序，即roll pitch yaw顺序
    Eigen::Vector3d transform=T.block(0,3,3,1);
    Eigen::Matrix<double,3,2> logM;
    logM.block(0,0,3,1)<<euler_angles;
    logM.block(0,1,3,1)<<transform;
    cout<<endl<<T<<endl;
}

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

#ifdef HAS_MRPT
#include <slamScene.h>
#endif

#include <stereoFrame.h>
#include <stereoFrameHandler.h>
#include <boost/filesystem.hpp>

#include <mapFeatures.h>
#include <mapHandler.h>

#include <dataset.h>
#include <timer.h>

using namespace StVO;
using namespace PLSLAM;

void showHelp();
bool getInputArgs(int argc, char **argv, std::string &dataset_name, int &frame_offset, int &frame_number, int &frame_step, std::string &config_file);

#define NO_SECENE

int main(int argc, char **argv)
{

    // read inputs
    string dataset_name, config_file;
    int frame_offset = 0, frame_number = 0, frame_step = 1;
    /*
     dataset_name 数据集名字 这里为zero
     frame_offset 跳过前多少个图像帧
     frame_number 只考虑这么多对帧
     frame_step   每多少个帧处理一次
     config_file 配置文件,默认配置为slamconfig
     */
    if (!getInputArgs(argc, argv, dataset_name, frame_offset, frame_number, frame_step, config_file)) {
        //参数检测错误时报错
        showHelp();
        return -1;
    }
    
    //显示传入参数
    cout << "" <<endl;
    for(int i=0;i<argc;i++)
        cout << i <<"  "<<argv[i] <<endl;
    
    //传入配置文件的数据
    if (!config_file.empty()) 
    {
        SlamConfig::loadFromFile(config_file);
        cout<<config_file<<endl;
    }

    //传入点词典
    if (SlamConfig::hasPoints() &&
            (!boost::filesystem::exists(SlamConfig::dbowVocP()) || !boost::filesystem::is_regular_file(SlamConfig::dbowVocP()))) {
        cout << "Invalid vocabulary for points" << endl;
        return -1;
    }

    //传入线词典
    if (SlamConfig::hasLines() &&
            (!boost::filesystem::exists(SlamConfig::dbowVocL()) || !boost::filesystem::is_regular_file(SlamConfig::dbowVocL()))) {
        cout << "Invalid vocabulary for lines" << endl;
        return -1;
    }

    // read dataset root dir fron environment variable
    boost::filesystem::path dataset_path(string( "/media/zhijian/Document/grow/slam/slamDataSet/KITTI/color"));
    
    if (!boost::filesystem::exists(dataset_path) || !boost::filesystem::is_directory(dataset_path)) {
        cout << "Check your DATASETS_DIR environment variable" << endl;
        return -1;
    }

    //添加具体数据集
    dataset_path /= dataset_name;
    if (!boost::filesystem::exists(dataset_path) || !boost::filesystem::is_directory(dataset_path)) {
        cout << "Invalid dataset path" << endl;
        return -1;
    }
    cout << endl << "Initializing PL-SLAM...." << flush;

    string dataset_dir = dataset_path.string();
    //传入相机配置文件，建立相机模型
    PinholeStereoCamera*  cam_pin = new PinholeStereoCamera((dataset_path / "dataset_params.yaml").string());
    //数据集类
    Dataset dataset(dataset_dir, *cam_pin, frame_offset, frame_number, frame_step);

    // create scene  std::string::npos 表示string最大值，表示不存在的未知
    string scene_cfg_name;
    if( (dataset_dir.find("KITTI")!=std::string::npos) ||
        (dataset_dir.find("malaga")!=std::string::npos)  )
        scene_cfg_name = "../config/scene_config.ini";
    else
        scene_cfg_name = "../config/scene_config_indoor.ini";
    //场景类
    #ifndef NO_SECENE
        slamScene scene(scene_cfg_name);
    #endif
    Matrix4d Tcw, Tfw = Matrix4d::Identity();
    Tcw = Matrix4d::Identity();
    
    #ifndef NO_SECENE
        scene.setStereoCalibration( cam_pin->getK(), cam_pin->getB() );
        scene.initializeScene(Tfw);
    #endif

    // create PLSLAM object
    PLSLAM::MapHandler* map = new PLSLAM::MapHandler(cam_pin);

    cout << " ... done. " << endl;

    Timer timer;

    // initialize and run PL-StVO
    int frame_counter = 0;
    StereoFrameHandler* StVO = new StereoFrameHandler(cam_pin);
    Mat img_l, img_r;
    ofstream fout("/media/zhijian/Document/grow/slam/slamDataSet/KITTI/data_odometry_poses/dataset/poses/my00.txt");
    //当数据集不空时
    while (dataset.nextFrame(img_l, img_r))
    {
        cout << "------------------------------------------   Frame #" << frame_counter
                 << "   ----------------------------------------" << endl;
        if( frame_counter == 0 ) // initialize
        {
            timer.start();
            //双目的初始化
            StVO->initialize(img_l,img_r,0);
            //第一帧肯定是关键帧
            PLSLAM::KeyFrame* kf = new PLSLAM::KeyFrame( StVO->prev_frame, 0 );
            //地图初始化
            map->initialize( kf );
            // update scene
            #ifndef NO_SECENE
                scene.initViewports( img_l.cols, img_r.rows );
                //更新raw image
                scene.setImage(StVO->prev_frame->plotStereoFrame());
                scene.updateSceneSafe( map );
            #endif
            StVO->T_w_curr=StVO->curr_frame->T_kf_f;
            cout << endl << "VO initialize: " << timer.stop() << endl;
        }
        else // run
        {
            // PL-StVO
            //frame_counter>0
            timer.start();
            StVO->insertStereoPair( img_l, img_r, frame_counter );
            //cout << endl << "VO insertStereoPair: " << timer.stop() << endl;
            timer.start();
            StVO->optimizePose();
            //cout << endl << "VO optimizePose: " << timer.stop() << endl;
            // check if a new keyframe is 
            #ifndef NO_SECENE
            scene.frame+=1;
            scene.setText(scene.frame,0,StVO->n_inliers_pt,StVO->n_inliers_ls);
            #endif
            if(map->prev_kf!=NULL)
                StVO->T_w_curr=map->prev_kf->T_w_kf*StVO->curr_frame->T_kf_f;
            else
                StVO->T_w_curr=StVO->curr_frame->T_kf_f;
            
            if( StVO->needNewKF() )
            {
//                 cout <<         "#KeyFrame:     " << map->max_kf_idx + 1;
//                 cout << endl << "#Points:       " << map->map_points.size();
//                 cout << endl << "#Segments:     " << map->map_lines.size();
//                 cout << endl << endl;

                // grab StF and update KF in StVO (the StVO thread can continue after this point)
                PLSLAM::KeyFrame* curr_kf = new PLSLAM::KeyFrame( StVO->curr_frame );
                // update KF in StVO
                StVO->currFrameIsKF();
                timer.start();
                map->addKeyFrame( curr_kf );
                //cout << endl << "VO addKeyFrame: " << timer.stop() << endl;
                // update scene
                #ifndef NO_SECENE
                    scene.setImage(StVO->curr_frame->plotStereoFrame());
                    scene.updateSceneSafe( map );
                #endif
            }
            else
            {
                #ifndef NO_SECENE
                    scene.setImage(StVO->curr_frame->plotStereoFrame());
                    scene.setPose( StVO->curr_frame->DT );
                    scene.updateScene();
                #endif
            }
            // update StVO
            StVO->updateFrame();
        }
        frame_counter++;
        vector<float> q = toQuaternion(StVO->T_w_curr.block(0,0,3,3));
        fout << setprecision(7) << " "<< StVO->T_w_curr(0, 3) << " " << StVO->T_w_curr(1, 3) << " " << StVO->T_w_curr(2, 3) << " "<< q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }
    fout.close();
    cout<<"over"<<endl;
    // finish SLAM
    map->finishSLAM();
    
    #ifndef NO_SECENE
        scene.updateScene( map );
    #endif
    // perform GBA
    cout << endl << "Performing Global Bundle Adjustment..." ;
    map->globalBundleAdjustment();
    cout << " ... done." << endl;
    #ifndef NO_SECENE
        scene.updateSceneGraphs( map );
    #endif

    // wait until the scene is closed
    #ifndef NO_SECENE
        while( scene.isOpen() );
    #endif

    return 0;
}

void showHelp() {
    cout << endl << "Usage: ./imgPLSLAM <dataset_name> [options]" << endl
         << "Options:" << endl
         << "\t-o Offset (number of frames to skip in the dataset directory" << endl
         << "\t-n Number of frames to process the sequence" << endl
         << "\t-s Parameter to skip s-1 frames (default 1)" << endl
         << "\t-c Config file" << endl
         << endl;
}

bool getInputArgs(int argc, char **argv, std::string &dataset_name, int &frame_offset, int &frame_number, int &frame_step, std::string &config_file) {

    if( argc < 2 || argc > 10 || (argc % 2) == 1 )
        return false;

    dataset_name = argv[1];
    int nargs = argc/2 - 1;
    for( int i = 0; i < nargs; i++ )
    {
        int j = 2*i + 2;
        if( string(argv[j]) == "-o" )
            frame_offset = stoi(argv[j+1]);
        else if( string(argv[j]) == "-n" )
            frame_number = stoi(argv[j+1]);
        else if( string(argv[j]) == "-s" )
            frame_step = stoi(argv[j+1]);
        else if (string(argv[j]) == "-c")
            config_file = string(argv[j+1]);
        else
            return false;
    }

    return true;
}

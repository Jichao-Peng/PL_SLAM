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
void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,vector<string> &vstrImageRight, vector<double> &vTimestamps);
//#define NO_SECENE

PinholeStereoCamera* Config::cam=NULL;//将相机参数传入配置文件

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
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestamps;
    LoadImages(dataset_dir, vstrImageLeft, vstrImageRight, vTimestamps);
    int nImages=vstrImageLeft.size();
    
    //传入相机配置文件，建立相机模型
    PinholeStereoCamera*  cam_pin = new PinholeStereoCamera((dataset_path / "dataset_params.yaml").string());
    Config::cam=cam_pin;
    //数据集类
    frame_offset=0;
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
    vector<vector<float>> stateTrack(nImages,vector<float>(6,0));
    
    // initialize and run PL-StVO
    int frame_counter = 0;
    float ninliers_ls=0.0;
    float ninliers_pt=0.0;
    StereoFrameHandler* StVO = new StereoFrameHandler(cam_pin);
    Mat img_l, img_r;
//     ofstream fout("/media/zhijian/Document/grow/slam/expriment/kitti/pl_slam/origin_00.txt");
    ofstream fout2("/media/zhijian/Document/grow/slam/slamDataSet/KITTI/data_odometry_poses/xyz.txt");
    //当数据集不空时
    //开始的时间
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    //提取特征
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    //估计位姿
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    //全部的时间
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    while (dataset.nextFrame(img_l, img_r))
    {
        frame_counter++;
        cout << "------------------------------------------   Frame #" << frame_counter
                 << "   ----------------------------------------" << endl;
        t1 = std::chrono::steady_clock::now();
        if( frame_counter == 1 ) // initialize
        {
            timer.start();
            //双目的初始化
            StVO->initialize(img_l,img_r,0);
            //第一帧肯定是关键帧
            PLSLAM::KeyFrame* kf = new PLSLAM::KeyFrame( StVO->prev_frame, 0 );
            //地图初始化
            map->initialize( kf );
            t2 = std::chrono::steady_clock::now();
            t3 = std::chrono::steady_clock::now();
            // update scene
            #ifndef NO_SECENE
                scene.initViewports( img_l.cols, img_r.rows );
                //更新raw image
                scene.setImage(StVO->prev_frame->plotStereoFrame());
                scene.updateSceneSafe( map );
            #endif
            StVO->T_w_curr=StVO->curr_frame->T_kf_f;
            StVO->n_inliers_pt=0;StVO->n_inliers_ls=0;
            stateTrack[frame_counter-1][5]=StVO->n_inliers_ls;
            cout << endl << "VO initialize: " << timer.stop() << endl;
            StVO->map_frames.push_back(new MapFrame(map->map_keyframes.size(),StVO->curr_frame->T_kf_f));
        }
        else // run
        {
            // PL-StVO
            timer.start();
            StVO->insertStereoPair( img_l, img_r, frame_counter-1 );
            t2 = std::chrono::steady_clock::now();
            //cout << endl << "VO insertStereoPair: " << timer.stop() << endl;
            timer.start();
            StVO->optimizePose();
            t3 = std::chrono::steady_clock::now();
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
            
            //看看每帧的线都是什么样子
            for (list<LineFeature*>::iterator it = StVO->matched_ls.begin(); it != StVO->matched_ls.end(); ++it)
                fout2<<(*it)->spl(0)<<" "<<(*it)->spl(1)<<" "<<(*it)->sdisp<<" "<<(*it)->epl(0)<<" "<<(*it)->epl(1)<<" "<<(*it)->edisp<<" ";
            fout2<<endl;
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
            StVO->map_frames.push_back(new MapFrame(map->map_keyframes.size(),StVO->curr_frame->T_kf_f));
            // update StVO
            StVO->updateFrame();
        }
        t4 = std::chrono::steady_clock::now();
//         ninliers_pt=(ninliers_pt*(frame_counter-1)+StVO->n_inliers_pt)/(frame_counter);
//         ninliers_ls=(ninliers_ls*(frame_counter-1)+StVO->n_inliers_ls)/(frame_counter);
        stateTrack[frame_counter-1][0]=std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        stateTrack[frame_counter-1][1]=std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t2).count();
        stateTrack[frame_counter-1][2]=std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t1).count();
        stateTrack[frame_counter-1][4]=StVO->n_inliers_pt;
        stateTrack[frame_counter-1][5]=StVO->n_inliers_ls;
        double T=0;
        if(frame_counter<nImages)
            T = vTimestamps[frame_counter]-vTimestamps[frame_counter-1];
        else if(frame_counter>0)
            T =  vTimestamps[frame_counter-1]-vTimestamps[frame_counter-2];
        stateTrack[frame_counter-1][3]=T;
        cout<<endl<<map->map_keyframes.size()<<'\t'<<StVO->map_frames.size()<<'\t'<<StVO->map_frames.back()->KfId<<endl;
    }
    // finish SLAM
    map->finishSLAM();
    
        // Tracking time statistics
    for(int kind=0;kind<6;kind++)
    {
        sort(stateTrack[kind].begin(),stateTrack[kind].end());
        float totaltime = 0;
        for(int ni=0; ni<nImages; ni++)
        {
            totaltime+=stateTrack[ni][kind];
        }
        switch (kind)
        {
            case 0:
                cout << "-------" << endl << endl;
                cout << "median detect time: " << stateTrack[nImages/2][kind] << endl;
                cout << "mean detect time: " << totaltime/nImages << endl;
                break;
            case 1:
                cout << "-------" << endl << endl;
                cout << "median optimizePose time: " << stateTrack[nImages/2][kind] << endl;
                cout << "mean optimizePose time: " << totaltime/nImages << endl;
                break;
            case 2:
                cout << "-------" << endl << endl;
                cout << "median tracking time: " << stateTrack[nImages/2][kind] << endl;
                cout << "mean tracking time: " << totaltime/nImages << endl;
                break;
            case 3:
                cout << "-------" << endl << endl;
                cout << "real median tracking time: " << stateTrack[nImages/2][kind] << endl;
                cout << "real mean tracking time: " << totaltime/nImages << endl;
                break;
            case 4:
                cout << "-------" << endl << endl;
                cout << "median n_inliers_pt: " << stateTrack[nImages/2][kind] << endl;
                cout << "mean n_inliers_pt: " << totaltime/nImages << endl;
                break;
            case 5:
                cout << "-------" << endl << endl;
                cout << "median n_inliers_ls: " << stateTrack[nImages/2][kind] << endl;
                cout << "mean n_inliers_ls: " << totaltime/nImages << endl;
                break;
        }
    }
    
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
    cout<<endl<<map->map_keyframes.size()<<'\t'<<StVO->map_frames.size()<<'\t'<<StVO->map_frames.back()->KfId<<endl;
//     string s;
//     transKitti(StVO->T_w_curr,s);
//     fout << s << endl;
//     fout.close();
    StVO->SaveTrajectoryKITTI("/media/zhijian/Document/grow/slam/expriment/kitti/pl_slam/origin_00.txt",map->map_keyframes);
    fout2.close();
    cout<<"over"<<endl;
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


void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";
    string strPrefixRight = strPathToSequence + "/image_1/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
    }
}

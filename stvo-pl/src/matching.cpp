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

#include "matching.h"

//STL
#include <cmath>
#include <functional>
#include <future>
#include <limits>
#include <stdexcept>

//OpenCV
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "config.h"
#include "gridStructure.h"

namespace StVO {

int matchNNR(const cv::Mat &desc1, const cv::Mat &desc2, float nnr, std::vector<int> &matches_12) {

    int matches = 0;
    matches_12.resize(desc1.rows, -1);

    std::vector<std::vector<cv::DMatch>> matches_;
    //①匹配描述子时，使用暴力匹配，Hamming距离；
    cv::Ptr<cv::BFMatcher> bfm = cv::BFMatcher::create(cv::NORM_HAMMING, false); // cross-check
    // 对每一个pdesc_1，在pdesc_2中寻找最近的2个描述子
    bfm->knnMatch(desc1, desc2, matches_, 2);

    if (desc1.rows != matches_.size())
        throw std::runtime_error("[matchNNR] Different size for matches and descriptors!");

    /*遍历匹配 pmatches_12，如果 pmatches_12 的询问点和  pmatches_21的训练点是一样的，并且 pmatches_12 的最佳匹配距离比次优匹配的距离大于Config::minRatio12P()，则认为这个点特征是内点，并把该点放到匹配点集 matched_pt中。*/
    for (int idx = 0; idx < desc1.rows; ++idx) {
        if (matches_[idx][0].distance < matches_[idx][1].distance * nnr) {
            matches_12[idx] = matches_[idx][0].trainIdx;
            matches++;
        }
    }

    return matches;
}

//nnr minRatio12P
//传入 上一帧，当前帧的描述子矩阵，参数，结果
int match(const cv::Mat &desc1, const cv::Mat &desc2, float nnr, std::vector<int> &matches_12) {

    //true if double-checking the matches between the two images
    if (Config::bestLRMatches()) {
        int matches;
        std::vector<int> matches_21;
        //是否同时点线
        if (Config::lrInParallel()) {
            /*std::ref 用于包装按引用传递的值。 std::cref 用于包装按const引用传递的值。*/
            auto match_12 = std::async(std::launch::async, &matchNNR,
                                  std::cref(desc1), std::cref(desc2), nnr, std::ref(matches_12));
            auto match_21 = std::async(std::launch::async, &matchNNR,
                                  std::cref(desc2), std::cref(desc1), nnr, std::ref(matches_21));
            matches = match_12.get();
            match_21.wait();
        } else {
            matches = matchNNR(desc1, desc2, nnr, matches_12);
            matchNNR(desc2, desc1, nnr, matches_21);
        }

        for (int i1 = 0; i1 < matches_12.size(); ++i1) {
            int &i2 = matches_12[i1];
            if (i2 >= 0 && matches_21[i2] != i1) {
                i2 = -1;
                matches--;
            }
        }

        return matches;
    } else
        return matchNNR(desc1, desc2, nnr, matches_12);
}

int distance(const cv::Mat &a, const cv::Mat &b) {

    // adapted from: http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel

    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist = 0;
    for(int i = 0; i < 8; i++, pa++, pb++) {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

//matchGrid(coords, pdesc_l, grid, pdesc_r, w, matches_12);
//此时左图像是栅格坐标点，右图像是每个栅格有哪些关键点
int matchGrid(const std::vector<point_2d> &points1, const cv::Mat &desc1, const GridStructure &grid, const cv::Mat &desc2, const GridWindow &w, std::vector<int> &matches_12) {

    if (points1.size() != desc1.rows)
        throw std::runtime_error("[matchGrid] Each point needs a corresponding descriptor!");

    int matches = 0;
    // 有描述子的行数那么多//关键点的个数和描述子的行数一样多，默认每个都是-1
    matches_12.resize(desc1.rows, -1);

    int best_d, best_d2, best_idx;
    //图像2（右图像）的匹配向量
    std::vector<int> matches_21, distances;

    if (Config::bestLRMatches()) {
        //默认值-1代表无匹配点
        matches_21.resize(desc2.rows, -1);
        //默认无穷远（int最大值）
        distances.resize(desc2.rows, std::numeric_limits<int>::max());
    }

    for (int i1 = 0; i1 < points1.size(); ++i1) {

        //最短距离和倒数第二短的距离
        best_d = std::numeric_limits<int>::max();
        best_d2 = std::numeric_limits<int>::max();
        best_idx = -1;

        //对于第i1个左关键点（栅格坐标）
        //points1就是之前算好的coords序列
        const std::pair<int, int> &coords = points1[i1];
        cv::Mat desc = desc1.row(i1);

        //取对应的右图像的栅格的关键点
        std::unordered_set<int> candidates;
        grid.get(coords.first, coords.second, w, candidates);

        if (candidates.empty()) continue;
        for (const int &i2 : candidates) {
            if (i2 < 0 || i2 >= desc2.rows) continue;
            //计算描述子距离
            const int d = distance(desc, desc2.row(i2));

            if (Config::bestLRMatches()) {
                if (d < distances[i2]) {
                    distances[i2] = d;
                    //matches_21中第i2个右关键点对应左图像的第i1个关键点
                    matches_21[i2] = i1;
                } else continue;
            }

            //更新best_d best_d2
            if (d < best_d) {
                best_d2 = best_d;
                best_d = d;
                best_idx = i2;
            } else if (d < best_d2)
                best_d2 = d;
        }

        //最短的比第二短的0.75还短，那更新左图像第i1个特征点对应的匹配点
        if (best_d < best_d2 * Config::minRatio12P()) {
            matches_12[i1] = best_idx;
            matches++;
        }
    }

    //最后对匹配结果进行遍历，如果不互为最有匹配，则放弃。  从算法来看，存在不互为匹配的情况？？？
    if (Config::bestLRMatches()) {
        for (int i1 = 0; i1 < matches_12.size(); ++i1) {
            int &i2 = matches_12[i1];
            if (i2 >= 0 && matches_21[i2] != i1) {
                i2 = -1;
                matches--;
            }
        }
    }

    //匹配点个数
    return matches;
}

int matchGrid(const std::vector<line_2d> &lines1, const cv::Mat &desc1,
              const GridStructure &grid, const cv::Mat &desc2, const std::vector<std::pair<double, double>> &directions2,
              const GridWindow &w,
              std::vector<int> &matches_12) {

    if (lines1.size() != desc1.rows)
        throw std::runtime_error("[matchGrid] Each line needs a corresponding descriptor!");

    int matches = 0;
    matches_12.resize(desc1.rows, -1);

    int best_d, best_d2, best_idx;
    std::vector<int> matches_21, distances;

    if (Config::bestLRMatches()) {
        matches_21.resize(desc2.rows, -1);
        distances.resize(desc2.rows, std::numeric_limits<int>::max());
    }

    for (int i1 = 0; i1 < lines1.size(); ++i1) {

        best_d = std::numeric_limits<int>::max();
        best_d2 = std::numeric_limits<int>::max();
        best_idx = -1;

        const line_2d &coords = lines1[i1];
        cv::Mat desc = desc1.row(i1);

        const point_2d sp = coords.first;
        const point_2d ep = coords.second;

        std::pair<double, double> v = std::make_pair(ep.first - sp.first, ep.second - sp.second);
        normalize(v);

        std::unordered_set<int> candidates;
        grid.get(sp.first, sp.second, w, candidates);
        grid.get(ep.first, ep.second, w, candidates);

        if (candidates.empty()) continue;
        for (const int &i2 : candidates) {
            if (i2 < 0 || i2 >= desc2.rows) continue;

            if (std::abs(dot(v, directions2[i2])) < Config::lineSimTh())
                continue;

            const int d = distance(desc, desc2.row(i2));

            if (Config::bestLRMatches()) {
                if (d < distances[i2]) {
                    distances[i2] = d;
                    matches_21[i2] = i1;
                } else continue;
            }

            if (d < best_d) {
                best_d2 = best_d;
                best_d = d;
                best_idx = i2;
            } else if (d < best_d2)
                best_d2 = d;
        }

        if (best_d < best_d2 * Config::minRatio12P()) {
            matches_12[i1] = best_idx;
            matches++;
        }
    }

    if (Config::bestLRMatches()) {
        for (int i1 = 0; i1 < matches_12.size(); ++i1) {
            int &i2 = matches_12[i1];
            if (i2 >= 0 && matches_21[i2] != i1) {
                i2 = -1;
                matches--;
            }
        }
    }

    return matches;
}

} //namesapce StVO

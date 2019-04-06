
#include "featureMatching.h"

//STL
#include <functional>
#include <future>
#include <stdexcept>
#include <vector>

//OpenCV
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "config.h"

namespace StVO {

//两个图像之间的匹配是不需要关键点的，只需要描述子，这里还是用的暴力匹配
int matchNNR(const cv::Mat &desc1, const cv::Mat &desc2, float nnr, std::vector<int> &matches_12) {

    int matches = 0;
    matches_12.resize(desc1.rows, -1);

    /*
     * DMatch
     queryIdx, trainIdx：指定了每幅图像中关键点与关键点列表中元素的匹配情况。其中，默认 query image 为新图片，而 training image 为旧图片
     imgIdx：指定要匹配哪一个训练图像
     distance：给出了匹配程度
     operator<()：给定基于 distance 的比较方式
     queryIdx : 查询点的索引（当前要寻找匹配结果的点在它所在图片上的索引）.
     trainIdx : 被查询到的点的索引（存储库中的点的在存储库上的索引）
     */
    std::vector<std::vector<cv::DMatch>> matches_;
    cv::Ptr<cv::BFMatcher> bfm = cv::BFMatcher::create(cv::NORM_HAMMING, false); // cross-check
    bfm->knnMatch(desc1, desc2, matches_, 2);

    if (desc1.rows != matches_.size())
        throw std::runtime_error("[MapHandler->matchNNR] Different size for matches and descriptors!");

    for (int idx = 0; idx < desc1.rows; ++idx) {
        //这里matches_[idx]储存着第idx个匹配点的（与所有候选点的）匹配距离序列
        if (matches_[idx][0].distance < matches_[idx][1].distance * nnr) {
            matches_12[idx] = matches_[idx][0].trainIdx;
            matches++;
        }
    }

    return matches;
}

int match(const cv::Mat &desc1, const cv::Mat &desc2, float nnr, std::vector<int> &matches_12) {

    matches_12.clear();
    if (Config::bestLRMatches()) {
        int matches;
        std::vector<int> matches_21;
        if (Config::lrInParallel()) {
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

} // namesapce StVO

#ifndef camFusion_hpp
#define camFusion_hpp


#include <stdio.h>
#include <vector>
#include <opencv2/core.hpp>
#include "dataStructures.h"


void clusterLidarWithROI(
  std::vector<BoundingBox>& boundingBoxes,
  std::vector<LidarPoint>&  lidarPoints,
  float                     shrinkFactor,
  cv::Mat&                  P_rect_xx,
  cv::Mat&                  R_rect_xx,
  cv::Mat&                  RT);

void clusterKptMatchesWithROI(
  BoundingBox&               boundingBox,
  std::vector<cv::KeyPoint>& kptsPrev,
  std::vector<cv::KeyPoint>& kptsCurr,
  std::vector<cv::DMatch>&   kptMatches);


/// @brief Method takes as input both the previous and the current data frames and
/// provides as output the ids of the matched regions of interest (i.e. the boxID property).
/// Matches must be the ones with the highest number of keypoint correspondences.
void matchBoundingBoxes(
  std::vector<cv::DMatch>& matches,
  std::map<int, int>&      bbBestMatches,
  DataFrame&               prevFrame,
  DataFrame&               currFrame);

void show3DObjects(
  std::vector<BoundingBox>& boundingBoxes,
  cv::Size                  worldSize,
  cv::Size                  imageSize,
  bool                      bWait=true);

void computeTTCCamera(
  std::vector<cv::KeyPoint>& kptsPrev,
  std::vector<cv::KeyPoint>& kptsCurr,
  std::vector<cv::DMatch>    kptMatches,
  double                     frameRate,
  double&                    TTC,
  cv::Mat*                   visImg=nullptr);

void computeTTCLidar(
  std::vector<LidarPoint>& lidarPointsPrev,
  std::vector<LidarPoint>& lidarPointsCurr,
  double                   frameRate,
  double&                  TTC);


/// @brief Sort Lidar points ascending on the x-coordinate
inline void sortLidarPointByX(std::vector<LidarPoint> &lidarPts) {
  std::sort(
    lidarPts.begin(),
    lidarPts.end(),
    [](LidarPoint a, LidarPoint b) { return a.x < b.x; }
    );
}
#endif // camFusion_hpp

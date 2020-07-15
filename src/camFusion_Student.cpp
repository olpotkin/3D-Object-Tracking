#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(
  std::vector<BoundingBox>& boundingBoxes,
  std::vector<LidarPoint>&  lidarPoints,
  float                     shrinkFactor,
  cv::Mat&                  P_rect_xx,
  cv::Mat&                  R_rect_xx,
  cv::Mat&                  RT)
{
  // Loop over all Lidar points and associate them to a 2D bounding box
  cv::Mat X(4, 1, cv::DataType<double>::type);
  cv::Mat Y(3, 1, cv::DataType<double>::type);

  for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1) {
    // Assemble vector for matrix-vector-multiplication
    X.at<double>(0, 0) = it1->x;
    X.at<double>(1, 0) = it1->y;
    X.at<double>(2, 0) = it1->z;
    X.at<double>(3, 0) = 1;

    // Project Lidar point into camera
    Y = P_rect_xx * R_rect_xx * RT * X;
    cv::Point pt;
    pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // Pixel coordinates
    pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

    // Pointers to all bounding boxes which enclose the current Lidar point
    std::vector<std::vector<BoundingBox>::iterator> enclosingBoxes;

    for (std::vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2) {
      // Shrink current bounding box slightly to avoid having too many outlier points around the edges
      cv::Rect smallerBox;
      smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
      smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
      smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
      smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

      // Check wether point is within current bounding box
      if (smallerBox.contains(pt)) {
        enclosingBoxes.push_back(it2);
      }

    } // eof loop over all bounding boxes

    // Check wether point has been enclosed by one or by multiple boxes
    if (enclosingBoxes.size() == 1) {
      // Add Lidar point to bounding box
      enclosingBoxes[0]->lidarPoints.push_back(*it1);
    }

  } // eof loop over all Lidar points
}


void show3DObjects(
  std::vector<BoundingBox>& boundingBoxes,
  cv::Size                  worldSize,
  cv::Size                  imageSize,
  bool                      bWait)
{
  // Create topview image
  cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

  for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1) {
    // Create randomized color for current 3D object
    cv::RNG rng(it1->boxID);
    cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

    // Plot Lidar points into top view image
    int top=1e8, left=1e8, bottom=0.0, right=0.0;
    float xwmin=1e8, ywmin=1e8, ywmax=-1e8;

    for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2) {
      // World coordinates
      float xw = (*it2).x; // World position in m with x facing forward from sensor
      float yw = (*it2).y; // World position in m with y facing left from sensor
      xwmin = xwmin<xw ? xwmin : xw;
      ywmin = ywmin<yw ? ywmin : yw;
      ywmax = ywmax>yw ? ywmax : yw;

      // Top-view coordinates
      int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
      int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

      // Find enclosing rectangle
      top    = top    < y ? top    : y;
      left   = left   < x ? left   : x;
      bottom = bottom > y ? bottom : y;
      right  = right  > x ? right  : x;

      // Draw individual point
      cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
    }

    // Draw enclosing rectangle
    cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

    // Augment object with some key data
    char str1[200], str2[200];
    sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
    putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
    sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
    putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);
  }

  // Plot distance markers
  float lineSpacing = 2.0; // Gap between distance markers
  int   nMarkers    = floor(worldSize.height / lineSpacing);

  for (size_t i = 0; i < nMarkers; ++i) {
    int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
    cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
  }

  // Display image
  std::string windowName = "3D Objects";
  cv::namedWindow(windowName, 1);
  cv::imshow(windowName, topviewImg);

  if (bWait) {
    // Wait for key to be pressed
    cv::waitKey(0);
  }
}


// Associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(
  BoundingBox&               boundingBox,
  std::vector<cv::KeyPoint>& kptsPrev,
  std::vector<cv::KeyPoint>& kptsCurr,
  std::vector<cv::DMatch>&   kptMatches)
{
  // Loop over all matches in the current frame
  for (cv::DMatch match : kptMatches) {
    if (boundingBox.roi.contains(kptsCurr[match.trainIdx].pt)) {
      boundingBox.kptMatches.push_back(match);
    }
  }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(
  std::vector<cv::KeyPoint>& kptsPrev,
  std::vector<cv::KeyPoint>& kptsCurr,
  std::vector<cv::DMatch>    kptMatches,
  double                     frameRate,
  double&                    TTC,
  cv::Mat*                   visImg)
{
  // Compute distance ratios on each pair of keypoints
  std::vector<double> distRatios;

  for (auto mIt_1 = kptMatches.begin(); mIt_1 != kptMatches.end() - 1; ++mIt_1) {
    // kptsCurr is indexed by trainIdx
    cv::KeyPoint kptOuterCurr = kptsCurr.at(mIt_1->trainIdx);
    // kptsPrev is indexed by queryIdx
    cv::KeyPoint kptOuterPrev = kptsPrev.at(mIt_1->queryIdx);

    for (auto mIt_2 = kptMatches.begin() + 1; mIt_2 != kptMatches.end(); ++mIt_2) {
      // kptsCurr is indexed by trainIdx
      cv::KeyPoint kpInnerCurr = kptsCurr.at(mIt_2->trainIdx);
      // kptsPrev is indexed by queryIdx
      cv::KeyPoint kpInnerPrev = kptsPrev.at(mIt_2->queryIdx);

      // Calculate the current and previous Euclidean distancesbetween keypoints
      double distanceCurr = cv::norm(kptOuterCurr.pt - kpInnerCurr.pt);
      double distancePrev = cv::norm(kptOuterPrev.pt - kpInnerPrev.pt);

      // Threshold
      double minDist = 100.0;

      // Avoid division by zero
      // Apply the threshold
      if (distancePrev > std::numeric_limits<double>::epsilon() && distanceCurr >= minDist) {
        double distRatio = distanceCurr / distancePrev;
        distRatios.push_back(distRatio);
      }
    }
  }

  // Continue if the vector of distRatios is not empty
  if (distRatios.empty()) {
    TTC = std::numeric_limits<double>::quiet_NaN();
    return;
  }

  // Median as a reasonable method of excluding outliers is used
  std::sort(distRatios.begin(), distRatios.end());
  double medianDistanceRatio = distRatios[distRatios.size() / 2];

  // Calculate a TTC estimate based on 2D camera features
  TTC = (-1.0 / frameRate) / (1 - medianDistanceRatio);
}


// Compute the time-to-collision for all matched 3D objects based on Lidar measurements alone
void computeTTCLidar(
  std::vector<LidarPoint>& lidarPointsPrev,
  std::vector<LidarPoint>& lidarPointsCurr,
  double                   frameRate,
  double&                  TTC)
{
  // In each frame, take the median x-distance
  sortLidarPointByX(lidarPointsPrev);
  sortLidarPointByX(lidarPointsCurr);

  // The previous frame's closing distance
  double dist_0 = lidarPointsPrev[lidarPointsPrev.size()/2].x;
  // The current frame's closing distance
  double dist_1 = lidarPointsCurr[lidarPointsCurr.size()/2].x;

  // NOTE: constant-velocity model is used
  // (1.0 / frameRate) - the time elapsed between images
  TTC = dist_1 * (1.0 / frameRate) / (dist_0 - dist_1);
}


void matchBoundingBoxes(
  std::vector<cv::DMatch>& matches,
  std::map<int, int>&      bbBestMatches,
  DataFrame&               prevFrame,
  DataFrame&               currFrame)
{
  int maxPrevBoxId = 0;
  std::multimap<int, int> mmStorage {};

  for (auto match : matches) {
    int prevBoxId = -1;
    int currBoxId = -1;

    cv::KeyPoint prevKp = prevFrame.keypoints[match.queryIdx];
    cv::KeyPoint currKp = currFrame.keypoints[match.trainIdx];

    // For each bounding box in the previous frame
    for (auto bBox : prevFrame.boundingBoxes) {
      if (bBox.roi.contains(prevKp.pt)) {
        prevBoxId = bBox.boxID;
      }
    }

    // For each bounding box in the current frame
    for (auto bBox : currFrame.boundingBoxes) {
      if (bBox.roi.contains(currKp.pt)) {
        currBoxId = bBox.boxID;
      }
    }

    // Add the containing boxID for each match to a multimap
    mmStorage.insert(std::make_pair(currBoxId, prevBoxId));

    maxPrevBoxId = std::max(maxPrevBoxId, prevBoxId);
  }

  // Setup a list of boxId's (int values) to iterate over in the current frame
  std::vector<int> currFrameBBoxIds {};

  for (const auto& bBox : currFrame.boundingBoxes) {
    currFrameBBoxIds.push_back(bBox.boxID);
  }

  // Loop through each bBoxId in the current frame,
  // and get the most frequent value of associated bBoxId for the previous frame
  for (int id : currFrameBBoxIds) {
    // Count the greatest number of matches in the multimap,
    // where each element has a [key=currBoxId] and [value=prevBoxId]
    //
    // std::multimap::equal_range(id) will return the range of all elements matching key = id.
    auto rangePrevBBoxIds = mmStorage.equal_range(id);

    // Create a vector of results (per current bounding box) of prevBBoxIds
    std::vector<int> results(maxPrevBoxId + 1, 0);

    // Accumulator loop
    for (auto it = rangePrevBBoxIds.first; it != rangePrevBBoxIds.second; ++it) {
      if (-1 != (*it).second){
        results[(*it).second] += 1;
      }
    }

    // Get the index of the maximum result of the previous frame's boxId
    int modeIdx = std::distance(results.begin(), std::max_element(results.begin(), results.end()));

    // Set the best matching bounding box map with
    // {key = Previous frame's most likely matching boxId, value = Current frame's boxId}
    bbBestMatches.insert(std::make_pair(modeIdx, id));
  }
}

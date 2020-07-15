#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"


// MAIN PROGRAM
int main(int argc, const char *argv[])
{
  // INIT VARIABLES AND DATA STRUCTURES

  // Data location
  std::string dataPath = "../";

  // Camera
  std::string imgBasePath = dataPath + "images/";
  // Left camera, color
  std::string imgPrefix   = "KITTI/2011_09_26/image_02/data/000000";
  std::string imgFileType = ".png";

  // First file index to load (assumes Lidar and camera names have identical naming convention)
  int imgStartIndex = 0;
  int imgEndIndex   = 18;  // Last file index to load
  int imgStepWidth  = 1;
  int imgFillWidth  = 4;   // No. of digits which make up the file index (e.g. img-0001.png)

  // Object detection
  std::string yoloBasePath           = dataPath     + "dat/yolo/";
  std::string yoloClassesFile        = yoloBasePath + "coco.names";
  std::string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
  std::string yoloModelWeights       = yoloBasePath + "yolov3.weights";

  // Lidar
  std::string lidarPrefix   = "KITTI/2011_09_26/velodyne_points/data/000000";
  std::string lidarFileType = ".bin";

  // Calibration data for camera and lidar
  cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
  cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
  cv::Mat RT(4,4,cv::DataType<double>::type);        // Rotation matrix and translation vector

  RT.at<double>(0,0) = 7.533745e-03;  RT.at<double>(0,1) = -9.999714e-01;
  RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
  RT.at<double>(1,0) = 1.480249e-02;  RT.at<double>(1,1) = 7.280733e-04;
  RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
  RT.at<double>(2,0) = 9.998621e-01;  RT.at<double>(2,1) = 7.523790e-03;
  RT.at<double>(2,2) = 1.480755e-02;  RT.at<double>(2,3) = -2.717806e-01;
  RT.at<double>(3,0) = 0.0;           RT.at<double>(3,1) = 0.0;
  RT.at<double>(3,2) = 0.0;           RT.at<double>(3,3) = 1.0;

  R_rect_00.at<double>(0,0) = 9.999239e-01;  R_rect_00.at<double>(0,1) = 9.837760e-03;
  R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
  R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01;
  R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
  R_rect_00.at<double>(2,0) = 7.402527e-03;  R_rect_00.at<double>(2,1) = 4.351614e-03;
  R_rect_00.at<double>(2,2) = 9.999631e-01;  R_rect_00.at<double>(2,3) = 0.0;
  R_rect_00.at<double>(3,0) = 0;             R_rect_00.at<double>(3,1) = 0;
  R_rect_00.at<double>(3,2) = 0;             R_rect_00.at<double>(3,3) = 1;

  P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00;
  P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
  P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02;
  P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
  P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00;
  P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;

  // Misc
  double sensorFrameRate = 10.0 / imgStepWidth; // Frames per second for Lidar and camera
  int    dataBufferSize  = 2;                   // No. of images which are held in memory (ring buffer) at the same time
  std::vector<DataFrame> dataBuffer;            // List of data frames which are held in memory at the same time
  bool              bVis = false;               // Visualize results

  /// MAIN LOOP OVER ALL IMAGES

  for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex += imgStepWidth) {
    /// LOAD IMAGE INTO BUFFER

    // Assemble filenames for current index
    std::ostringstream imgNumber;
    imgNumber << std::setfill('0') << std::setw(imgFillWidth) << imgStartIndex + imgIndex;
    std::string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

    // Load image from file
    cv::Mat img = cv::imread(imgFullFilename);

    // Push image into data frame buffer
    DataFrame frame;
    frame.cameraImg = img;
    dataBuffer.push_back(frame);

    std::cout << "#1 : LOAD IMAGE INTO BUFFER done" << std::endl;

    /// DETECT & CLASSIFY OBJECTS

    float confThreshold = 0.2;
    float nmsThreshold  = 0.4;

    detectObjects(
      (dataBuffer.end() - 1)->cameraImg,
      (dataBuffer.end() - 1)->boundingBoxes,
      confThreshold,
      nmsThreshold,
      yoloBasePath,
      yoloClassesFile,
      yoloModelConfiguration,
      yoloModelWeights,
      bVis);

    std::cout << "#2 : DETECT & CLASSIFY OBJECTS done" << std::endl;

    /// CROP LIDAR POINTS

    // Load 3D Lidar points from file
    std::string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
    std::vector<LidarPoint> lidarPoints;
    loadLidarFromFile(lidarPoints, lidarFullFilename);

    // Remove Lidar points based on distance properties
    // Focus on ego lane
    float minZ = -1.5, maxZ = -0.9;
    float minX = 2.0,  maxX = 20.0;
    float maxY = 2.0;
    float minR = 0.1;

    cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);

    (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

    std::cout << "#3 : CROP LIDAR POINTS done" << std::endl;

    /// CLUSTER LIDAR POINT CLOUD

    // Associate Lidar points with camera-based ROI
    // Shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
    float shrinkFactor = 0.10;
    clusterLidarWithROI(
      (dataBuffer.end()-1)->boundingBoxes,
      (dataBuffer.end() - 1)->lidarPoints,
      shrinkFactor,
      P_rect_00,
      R_rect_00,
      RT);

    // Visualize 3D objects
    bVis = true;
    if (bVis) {
      show3DObjects(
        (dataBuffer.end()-1)->boundingBoxes,
        cv::Size(4.0, 20.0),
        cv::Size(2000, 2000),
        true);
    }
    bVis = false;

    std::cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << std::endl;

    // REMOVE THIS LINE BEFORE PROCEEDING WITH THE FINAL PROJECT
    //continue; // Skips directly to the next image without processing what comes beneath

    /// DETECT IMAGE KEYPOINTS

    // Convert current image to grayscale
    cv::Mat imgGray;
    cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

    // Extract 2D keypoints from current image
    // Create empty feature list for current image
    std::vector<cv::KeyPoint> keypoints;

    //std::string detectorType = "SHITOMASI";
    //std::string detectorType = "HARRIS";
    std::string detectorType = "FAST";
    //std::string detectorType = "BRISK";
    //std::string detectorType = "ORB";
    //std::string detectorType = "AKAZE";
    //std::string detectorType = "SIFT";


    // SHI-TOMASI
    if (detectorType.compare("SHITOMASI") == 0) {
      detKeypointsShiTomasi(keypoints, imgGray, false);
    }
    // HARRIS
    else if (detectorType.compare("HARRIS") == 0) {
      detKeypointsHarris(keypoints, imgGray, false);
    }
    else if (detectorType.compare("FAST")  == 0 ||
             detectorType.compare("BRISK") == 0 ||
             detectorType.compare("ORB")   == 0 ||
             detectorType.compare("AKAZE") == 0 ||
             detectorType.compare("SIFT")  == 0) {
      detKeypointsModern(keypoints, imgGray, detectorType, false);
    }
    else {
      throw std::invalid_argument(detectorType + " IS NOT VALID");
    }

    // Optional : limit number of keypoints (helpful for debugging and learning)
    bool bLimitKpts = false;
    if (bLimitKpts) {
      int maxKeypoints = 50;

      if (detectorType.compare("SHITOMASI") == 0) {
        // There is no response info, so keep the first 50 as they are sorted in descending quality order
        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
      }
      cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
      std::cout << " NOTE: Keypoints have been limited!" << std::endl;
    }

    // Push keypoints and descriptor for current frame to end of data buffer
    (dataBuffer.end() - 1)->keypoints = keypoints;

    std::cout << "#5 : DETECT KEYPOINTS done" << std::endl;

    /// EXTRACT KEYPOINT DESCRIPTORS

    cv::Mat descriptors;

    //std::string descriptorType = "BRISK";
    //std::string descriptorType = "BRIEF";
    std::string descriptorType = "ORB";
    //std::string descriptorType = "FREAK";
    //std::string descriptorType = "AKAZE";  // Not compatible with non-AKAZE detectors
    //std::string descriptorType = "SIFT";   // Not compatible with ORB detectors

    descKeypoints(
      (dataBuffer.end() - 1)->keypoints,
      (dataBuffer.end() - 1)->cameraImg,
      descriptors,
      descriptorType);

    // Push descriptors for current frame to end of data buffer
    (dataBuffer.end() - 1)->descriptors = descriptors;

    std::cout << "#6 : EXTRACT DESCRIPTORS done" << std::endl;

    // Wait until at least two images have been processed
    if (dataBuffer.size() > 1) {
      /// MATCH KEYPOINT DESCRIPTORS

      std::vector<cv::DMatch> matches;

      // Brute force or Fast Library for Approximate Nearest Neighbors (FLANN)
      std::string matcherType = "MAT_BF";
      // string matcherType = "MAT_FLANN";

      // For descriptor type, select binary (BINARY) or histogram of gradients (HOG)
      // BINARY descriptors: BRISK, BRIEF, ORB, FREAK, and (A)KAZE.
      // HOG descriptors: SIFT (SURF, GLOH - patented).
      std::string descriptorCategory {};

      if (descriptorType == "SIFT") {
        descriptorCategory = "DES_HOG";
      }
      else {
        descriptorCategory = "DES_BINARY";
      }

      // Nearest neighbors (NN) or k nearest neighbors (KNN) for selector type
      // string selectorType = "SEL_NN";
      std::string selectorType = "SEL_KNN";

      matchDescriptors((dataBuffer.end() - 2)->keypoints,   (dataBuffer.end() - 1)->keypoints,
                       (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                       matches, descriptorCategory, matcherType, selectorType);

      // Store matches in current data frame
      (dataBuffer.end() - 1)->kptMatches = matches;

      std::cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << std::endl;

      /// TRACK 3D OBJECT BOUNDING BOXES

      /// STUDENT ASSIGNMENT
      /// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>)
      /// between current and previous frame (implement ->matchBoundingBoxes)
      std::map<int, int> bbBestMatches;
      // Associate bounding boxes between current and previous frame using keypoint matches
      matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1));
      // Trace data (for debugging)
      //for (auto bbKey : bbBestMatches) {
      //  std::cout << "{bbKey = " << bbKey.first << ", bbValue=" << bbKey.second << "}" << std::endl;
      //}
      /// EOF STUDENT ASSIGNMENT

      // Store matches in current data frame
      (dataBuffer.end()-1)->bbMatches = bbBestMatches;

      std::cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << std::endl;

      /// COMPUTE TTC ON OBJECT IN FRONT

      // Loop over all BB match pairs
      for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin();
                it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1) {
        // Find bounding boxes associates with current match
        BoundingBox *prevBB, *currBB;

        for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin();
                  it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2) {
          // Check wether current match partner corresponds to this BB
          if (it1->second == it2->boxID) {
            currBB = &(*it2);
          }
        }

        for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin();
                  it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2) {
          // Check wether current match partner corresponds to this BB
          if (it1->first == it2->boxID) {
            prevBB = &(*it2);
          }
        }

        // Compute TTC for current match
        // Only compute TTC if we have Lidar points
        if (currBB->lidarPoints.size() > 0 && prevBB->lidarPoints.size() > 0)
        {
          /// STUDENT ASSIGNMENT
          /// TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
          double ttcLidar = 0.0;
          computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar);
          /// EOF STUDENT ASSIGNMENT

          /// STUDENT ASSIGNMENT
          /// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
          /// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)
          double ttcCamera = 0.0;

          clusterKptMatchesWithROI(*currBB,
            (dataBuffer.end() - 2)->keypoints,
            (dataBuffer.end() - 1)->keypoints,
            (dataBuffer.end() - 1)->kptMatches);

          computeTTCCamera(
            (dataBuffer.end() - 2)->keypoints,
            (dataBuffer.end() - 1)->keypoints,
            currBB->kptMatches, sensorFrameRate, ttcCamera);
          /// EOF STUDENT ASSIGNMENT

          bVis = true;
          if (bVis) {
            cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();

            showLidarTopview(currBB->lidarPoints, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true);

            showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);

            cv::rectangle(visImg,
              cv::Point(currBB->roi.x, currBB->roi.y),
              cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height),
              cv::Scalar(0, 255, 0), 2);

            char str[200];
            sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
            putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));

            std::string windowName = "Final Results : TTC";
            cv::namedWindow(windowName, 4);
            cv::imshow(windowName, visImg);
            std::cout << "Press key to continue to next frame" << std::endl;
            cv::waitKey(0);
          }
          bVis = false;

        } // eof TTC computation
      } // eof loop over all BB matches

    }

  } // eof loop over all images

  return 0;
}

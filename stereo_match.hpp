#include <stdio.h>

#include <boost/format.hpp>
#include <iostream>
#include <memory>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
using namespace cv;
using namespace std;

//! stereo camera match
class StereoMatch {
 public:
  StereoMatch();

  // ~StereoMatch();

  enum {
    STEREO_BM = 0,
    STEREO_SGBM = 1,
    STEREO_HH = 2,
    STEREO_VAR = 3,
    STEREO_3WAY = 4
  };
  int feature_method = 1;
  //! Intrinsic parameters
  Mat M1, D1, M2, D2;
  //! Extrinsic parameters
  Mat R, T, R1, P1, R2, P2;
  //! disparity map and depth map
  cv::Mat disp, disp8, depth;
  Mat pic_l, pic_r;
  vector<KeyPoint> keypoints1, keypoints2;
  Mat descriptors1, descriptors2;
  typedef struct {
    Point2f left_top;
    Point2f left_bottom;
    Point2f right_top;
    Point2f right_bottom;
  } four_corners_t;

  four_corners_t corners;
  Mat camera_matrix;
  Mat distortion_coefficients;
  Mat transH;
  void Init(int choice);

  void Process(cv::Mat img_l, cv::Mat img_r);

  void disp2Depth(cv::Mat dispMap, cv::Mat& depthMap, cv::Mat K);

  void insertDepth32f(cv::Mat& depth);

  void featurematch(Mat src1, Mat src2);
  void CalcCorners(const Mat& H, const Mat& src);
  void stitchImage(Mat src1, Mat src2);
  void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst);
};

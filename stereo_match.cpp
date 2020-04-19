#include "stereo_match.hpp"
enum {
  STEREO_BM = 0,
  STEREO_SGBM = 1,
  STEREO_HH = 2,
  STEREO_VAR = 3,
  STEREO_3WAY = 4
};
StereoMatch::StereoMatch() {}

void StereoMatch::Init(int choice) {
  FileStorage file_storage("/home/gjx/opencv/open/stereo_camera/new.xml",
                           FileStorage::READ);
  file_storage["camera_matrix"] >> camera_matrix;
  cout << "ready" << endl;
  file_storage["distortion_coefficients"] >> distortion_coefficients;
  file_storage["rotate"] >> R;
  cout << R << endl;
  file_storage["translation"] >> T;
  cout << T << endl;
  if (choice == 0) {
    M1 = camera_matrix;
    M2 = camera_matrix;
    D1 = distortion_coefficients;
    D2 = distortion_coefficients;
    this->R = R;
    this->T = T;
  } else {
    M1 = camera_matrix;
    M2 = camera_matrix;
    D1 = distortion_coefficients;
    D2 = distortion_coefficients;
    cout << M1 << endl;
  }
  file_storage.release();
}

void StereoMatch::CalcCorners(const Mat& H, const Mat& src) {
  double v2[] = {0, 0, 1};           //左上角
  double v1[3];                      //变换后的坐标值
  Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
  Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量

  V1 = H * V2;
  //左上角(0,0,1)
  // cout << "V2: " << V2 << endl;
  // cout << "V1: " << V1 << endl;
  corners.left_top.x = v1[0] / v1[2];
  corners.left_top.y = v1[1] / v1[2];

  //左下角(0,src.rows,1)
  v2[0] = 0;
  v2[1] = src.rows;
  v2[2] = 1;
  V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
  V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
  V1 = H * V2;
  corners.left_bottom.x = v1[0] / v1[2];
  corners.left_bottom.y = v1[1] / v1[2];

  //右上角(src.cols,0,1)
  v2[0] = src.cols;
  v2[1] = 0;
  v2[2] = 1;
  V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
  V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
  V1 = H * V2;
  corners.right_top.x = v1[0] / v1[2];
  corners.right_top.y = v1[1] / v1[2];

  //右下角(src.cols,src.rows,1)
  v2[0] = src.cols;
  v2[1] = src.rows;
  v2[2] = 1;
  V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
  V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
  V1 = H * V2;
  corners.right_bottom.x = v1[0] / v1[2];
  corners.right_bottom.y = v1[1] / v1[2];
}

//优化两图的连接处，使得拼接自然
void StereoMatch::OptimizeSeam(Mat& img1, Mat& trans, Mat& dst) {
  int start = MIN(corners.left_top.x,
                  corners.left_bottom.x);  //开始位置，即重叠区域的左边界

  double processWidth = img1.cols - start;  //重叠区域的宽度
  int rows = dst.rows;
  int cols = img1.cols;  //注意，是列数*通道数
  double alpha = 1;      // img1中像素的权重
  for (int i = 0; i < rows; i++) {
    uchar* p = img1.ptr<uchar>(i);  //获取第i行的首地址
    uchar* t = trans.ptr<uchar>(i);
    uchar* d = dst.ptr<uchar>(i);
    for (int j = start; j < cols; j++) {
      //如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
      if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0) {
        alpha = 1;
      } else {
        // img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好
        alpha = (processWidth - (j - start)) / processWidth;
      }

      d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
      d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
      d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);
    }
  }
}

void StereoMatch::featurematch(Mat src1, Mat src2) {
  // src1:right src2:left
  vector<DMatch> matches;
  if (feature_method == 0) {
    Ptr<Feature2D> surf = xfeatures2d::SURF::create();

    surf->detectAndCompute(src1, Mat(), keypoints1, descriptors1);
    surf->detectAndCompute(src2, Mat(), keypoints2, descriptors2);
    // drawKeypoints(src1,keypoints1,src1);

    FlannBasedMatcher
        matcher;  //不使用暴力匹配，改成Fast Library for Approximate Nearest
                  // Neighbors匹配（近似算法，比暴力匹配更快）

    matcher.match(descriptors1, descriptors2, matches);
  } else if (feature_method == 1) {
    // 1 初始化特征点和描述子,ORB

    Ptr<ORB> orb = ORB::create();

    // 2 提取 Oriented FAST 特征点
    orb->detect(src1, keypoints1);
    orb->detect(src2, keypoints2);

    // 3 根据角点位置计算 BRIEF 描述子
    orb->compute(src1, keypoints1, descriptors1);
    orb->compute(src2, keypoints2, descriptors2);

    // 4 对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离

    BFMatcher bfmatcher(NORM_HAMMING);
    bfmatcher.match(descriptors1, descriptors2, matches);

    // 5 匹配对筛选
    double min_dist = 1000, max_dist = 0;
    // 找出所有匹配之间的最大值和最小值
    for (int i = 0; i < descriptors1.rows; i++) {
      double dist = matches[i].distance;  //汉明距离在matches中
      if (dist < min_dist) min_dist = dist;
      if (dist > max_dist) max_dist = dist;
    }
    // 当描述子之间的匹配大于2倍的最小距离时，即认为该匹配是一个错误的匹配。
    // 但有时描述子之间的最小距离非常小，可以设置一个经验值作为下限
    vector<DMatch> good_matches;
    for (int i = 0; i < descriptors1.rows; i++) {
      if (matches[i].distance <= max(2 * min_dist, 30.0))
        good_matches.push_back(matches[i]);
    }
    matches.clear();
    matches = good_matches;
  }

  vector<Point2f> pic1, pic2;  //滤掉误匹配点
  for (int i = 0; i < matches.size(); i++) {
    pic1.push_back(keypoints1[matches[i].queryIdx].pt);
    pic2.push_back(keypoints2[matches[i].trainIdx].pt);
  }
  vector<unsigned char> mark(pic1.size());
  transH = findHomography(pic1, pic2, CV_RANSAC, 5, mark, 500);
  Mat E = cv::findEssentialMat(pic1, pic2, camera_matrix, CV_RANSAC);
  cv::Mat R1, R2, t;
  cv::decomposeEssentialMat(E, R1, R2, t);
  // decomposeHomographyMat(transM, camera_matrix, R3, t3, noArray());
  this->R = R1.clone();
  this->T = -t.clone();
  cout << "R" << R << endl;
  cout << "T" << T << endl;
}

void StereoMatch::stitchImage(Mat src1, Mat src2) {
  Mat tempP, dst1, dst2;
  CalcCorners(transH, src1);
  warpPerspective(src1, tempP, transH,
                  Size(MIN(corners.right_top.x, corners.right_bottom.x),
                       MAX(src2.rows, src1.rows)));
  Mat matchP(tempP.cols, tempP.rows, CV_8UC3);
  tempP.copyTo(matchP);
  // cout << "src2" << src2.cols << "   " << src2.rows << "temp" << tempP.cols
  //      << "   " << tempP.rows << endl;
  src2.copyTo(matchP(Rect(0, 0, src2.cols, src2.rows)));
  // cv::imshow("compare", tempP);
  // cv::imshow("compare1", matchP);
  dst2 = src2.clone();
  OptimizeSeam(dst2, tempP, matchP);
  resize(matchP, matchP, Size(matchP.cols * 0.2, matchP.rows * 0.2));
  imshow("result", matchP);
}

void StereoMatch::Process(cv::Mat img_l, cv::Mat img_r) {
  int method = STEREO_SGBM;
  int SADWindowSize, numberOfDisparities;
  bool no_display;
  float scale;

  Ptr<StereoBM> bm = StereoBM::create(16, 9);
  Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16, 3);

  Size img_size = img_l.size();

  Rect roi1, roi2;
  cv::Mat Q;
  cout << "start" << endl;
  //图像矫正 摆正 计算
  cv::stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2,
                    Q);  //,
                         // CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);
  cout << "stereosuccess" << endl;
  cv::Mat map11, map12, map21, map22;
  cv::initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_32FC1, map11, map12);
  cv::initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_32FC1, map21, map22);
  //图像矫正
  cout << "map" << endl;
  cv::Mat img1r, img2r;
  cv::remap(img_l, img1r, map11, map12, INTER_LINEAR);
  cv::remap(img_r, img2r, map21, map22, INTER_LINEAR);
  img_l = img1r;
  img_r = img2r;

  numberOfDisparities = numberOfDisparities > 0
                            ? numberOfDisparities
                            : ((img_size.width / 8) + 15) & -16;
  cout << "method" << endl;
  switch (method) {
    case STEREO_BM:
      // method 1:BM
      bm->setROI1(roi1);
      bm->setROI2(roi2);
      bm->setPreFilterCap(31);
      bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 9);
      bm->setMinDisparity(
          0);  //是控制匹配搜索的第一个参数，代表了匹配搜苏从哪里开始
      bm->setNumDisparities(numberOfDisparities);  //表示最大搜索视差数
      bm->setTextureThreshold(10);
      bm->setUniquenessRatio(15);  //表示匹配功能函数
      bm->setSpeckleWindowSize(100);
      bm->setSpeckleRange(32);
      bm->setDisp12MaxDiff(1);
      break;
    case STEREO_SGBM:
      // method 2:SBGM
      sgbm->setPreFilterCap(63);
      int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
      sgbm->setBlockSize(sgbmWinSize);

      int cn = img_l.channels();

      sgbm->setP1(8 * cn * sgbmWinSize * sgbmWinSize);
      sgbm->setP2(32 * cn * sgbmWinSize * sgbmWinSize);
      sgbm->setMinDisparity(
          0);  //是控制匹配搜索的第一个参数，代表了匹配搜索从哪里开始
      sgbm->setNumDisparities(numberOfDisparities);  //表示最大搜索视差数
      sgbm->setUniquenessRatio(10);                  //表示匹配功能函数
      sgbm->setSpeckleWindowSize(100);
      sgbm->setSpeckleRange(32);
      sgbm->setDisp12MaxDiff(1);
      if (method == STEREO_HH)
        sgbm->setMode(StereoSGBM::MODE_HH);
      else if (method == STEREO_SGBM)
        sgbm->setMode(StereoSGBM::MODE_SGBM);
      else if (method == STEREO_3WAY)
        sgbm->setMode(StereoSGBM::MODE_SGBM_3WAY);
      break;
  }
  cout << "666" << endl;
  // Mat img1p, img2p, dispp;
  // copyMakeBorder(img1, img1p, 0, 0, numberOfDisparities, 0,
  // IPL_BORDER_REPLICATE); copyMakeBorder(img2, img2p, 0, 0,
  // numberOfDisparities, 0, IPL_BORDER_REPLICATE);

  int64 t = getTickCount();
  if (method == STEREO_BM)
    bm->compute(img_l, img_r, disp);
  else if (method == STEREO_SGBM || method == STEREO_HH ||
           method == STEREO_3WAY)
    sgbm->compute(img_l, img_r, disp);
  t = getTickCount() - t;
  printf("Time elapsed: %fms\n", t * 1000 / getTickFrequency());
  resize(disp, disp, Size(disp.cols * 0.2, disp.rows * 0.2));
  cv::imshow("disp", disp);
  // disp = dispp.colRange(numberOfDisparities, img_lp.cols);
  // if (method != STEREO_VAR)
  //   disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities * 16.));
  // else
  //   disp.convertTo(disp8, CV_8U);

  // // disp to depth
  // disp2Depth(disp8, depth, M1);
  // if (!no_display) {
  //   namedWindow("left", 1);
  //   imshow("left", img_l);
  //   namedWindow("right", 1);
  //   imshow("right", img_r);
  //   namedWindow("disparity", 0);
  //   imshow("disparity", disp8);
  //   printf("press any key to continue...");
  //   fflush(stdout);
  //   waitKey();
  //   printf("\n");
  // }
}

void StereoMatch::disp2Depth(cv::Mat dispMap, cv::Mat& depthMap, cv::Mat K) {
  int type = dispMap.type();

  float fx = K.at<float>(0, 0);
  float fy = K.at<float>(1, 1);
  float cx = K.at<float>(0, 2);
  float cy = K.at<float>(1, 2);
  float baseline = 65;  // baseline

  if (type == CV_8U) {
    const float PI = 3.14159265358;
    int height = dispMap.rows;
    int width = dispMap.cols;

    uchar* dispData = (uchar*)dispMap.data;
    ushort* depthData = (ushort*)depthMap.data;
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        int id = i * width + j;
        if (!dispData[id]) continue;
        depthData[id] = ushort((float)fx * baseline / ((float)dispData[id]));
      }
    }
  } else {
    std::cout << "please confirm dispImg's type!" << std::endl;
    cv::waitKey(0);
  }
}

//空洞填充
void StereoMatch::insertDepth32f(cv::Mat& depth) {
  const int width = depth.cols;
  const int height = depth.rows;
  float* data = (float*)depth.data;
  cv::Mat integralMap = cv::Mat::zeros(height, width, CV_64F);
  cv::Mat ptsMap = cv::Mat::zeros(height, width, CV_32S);
  double* integral = (double*)integralMap.data;
  int* ptsIntegral = (int*)ptsMap.data;
  memset(integral, 0, sizeof(double) * width * height);
  memset(ptsIntegral, 0, sizeof(int) * width * height);
  for (int i = 0; i < height; ++i) {
    int id1 = i * width;
    for (int j = 0; j < width; ++j) {
      int id2 = id1 + j;
      if (data[id2] > 1e-3) {
        integral[id2] = data[id2];
        ptsIntegral[id2] = 1;
      }
    }
  }
  // 积分区间
  for (int i = 0; i < height; ++i) {
    int id1 = i * width;
    for (int j = 1; j < width; ++j) {
      int id2 = id1 + j;
      integral[id2] += integral[id2 - 1];
      ptsIntegral[id2] += ptsIntegral[id2 - 1];
    }
  }
  for (int i = 1; i < height; ++i) {
    int id1 = i * width;
    for (int j = 0; j < width; ++j) {
      int id2 = id1 + j;
      integral[id2] += integral[id2 - width];
      ptsIntegral[id2] += ptsIntegral[id2 - width];
    }
  }
  int wnd;
  double dWnd = 2;
  while (dWnd > 1) {
    wnd = int(dWnd);
    dWnd /= 2;
    for (int i = 0; i < height; ++i) {
      int id1 = i * width;
      for (int j = 0; j < width; ++j) {
        int id2 = id1 + j;
        int left = j - wnd - 1;
        int right = j + wnd;
        int top = i - wnd - 1;
        int bot = i + wnd;
        left = max(0, left);
        right = min(right, width - 1);
        top = max(0, top);
        bot = min(bot, height - 1);
        int dx = right - left;
        int dy = (bot - top) * width;
        int idLeftTop = top * width + left;
        int idRightTop = idLeftTop + dx;
        int idLeftBot = idLeftTop + dy;
        int idRightBot = idLeftBot + dx;
        int ptsCnt = ptsIntegral[idRightBot] + ptsIntegral[idLeftTop] -
                     (ptsIntegral[idLeftBot] + ptsIntegral[idRightTop]);
        double sumGray = integral[idRightBot] + integral[idLeftTop] -
                         (integral[idLeftBot] + integral[idRightTop]);
        if (ptsCnt <= 0) {
          continue;
        }
        data[id2] = float(sumGray / ptsCnt);
      }
    }
    int s = wnd / 2 * 2 + 1;
    if (s > 201) {
      s = 201;
    }
    cv::GaussianBlur(depth, depth, cv::Size(s, s), s, s);
  }
}
int main() {
  Mat src = imread("/home/gjx/opencv/open/stereo_camera/2.jpg");
  Mat distortion = src.clone();
  Mat camera_matrix = Mat(3, 3, CV_32FC1);
  Mat distortion_coefficients;

  clock_t startTime, endTime;

  startTime = clock();
  vector<Mat> srcs;
  // left
  Mat src1 = imread("/home/gjx/opencv/open/stereo_camera/homography/1.jpg");
  // right
  Mat src2 = imread("/home/gjx/opencv/open/stereo_camera/homography/2.jpg");

  if (src1.data == NULL || src2.data == NULL) {
    cout << "No exist" << endl;
    return -1;
  }

  StereoMatch st;
  cout << "param init" << endl;
  st.Init(1);
  st.featurematch(src2, src1);
  // st.stitchImage(src2, src1);
  // endTime = clock();  //计时结束
  // cout << "The run time is: " << (double)(endTime - startTime) /
  // CLOCKS_PER_SEC
  //  << "s" << endl;

  st.Process(src1, src2);
  waitKey(0);
}

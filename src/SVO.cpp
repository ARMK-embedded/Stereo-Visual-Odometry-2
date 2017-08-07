
/*#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include <iostream>*/

#include "SVO.h"

using namespace cv;
using namespace std;
using namespace Eigen;

#pragma once

SVO::SVO() // default constructor
{

}
SVO::~SVO() // default destructor
{

}


void SVO::Xinit()
{
  X = VectorXd::Zero(6,1);
  P = MatrixXd::Zero(6,6);
  H = MatrixXd::Zero(3,6);
  Q = 0.001*Matrix3d::Identity();
  POSE = MatrixXd::Identity(4,4);
  imgCount = 0;
}

void SVO::removeMatches(vector<DMatch> &matches)
{
// Remove repeated matches
    for(int i = 0; i < matches.size(); i++){

      bool repeated = false;
      for(int j = i+1; j < matches.size(); j++){

        if(matches[i].trainIdx == matches[j].trainIdx){
          repeated = true;
          matches.erase(matches.begin() + j);
          j--;
        }
      }


      if(repeated)
        matches.erase(matches.begin() + i);
    }

}

void Rectification(Mat &img1, Mat &img2, Mat &M1, Mat &M2, Mat &D1, Mat &D2, Mat &T)
{
    Mat R = Mat::eye(3,3,CV_64F);
    Size img_size = img1.size();
    Rect roi1, roi2;
    Mat R1, P1, R2, P2;
    Mat Q;
                                                                                            //alpha
    stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY,  0, img_size, &roi1, &roi2 );

    Mat map11, map12, map21, map22;
    initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
    initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

    Mat img1r, img2r;
    remap(img1, img1r, map11, map12, INTER_LINEAR);
    remap(img2, img2r, map21, map22, INTER_LINEAR);

    img1 = img1r;
    img2 = img2r;


}


void draw3DPoints(Mat &img, vector<Point2f> &pts2D)
{
  for (int i = 0; i < (int)pts2D.size(); i++)
  {
     
          Point2f pt = pts2D[i];
          cv::circle(img, pt, 2, Scalar(255, 0, 0), 1);


  }


}


//Copy (x,y) location of descriptor matches found from KeyPoint data structures into Point2f vectors
static void matches2points(const vector<DMatch>& matches, const vector<KeyPoint>& kpts_train,
                    const vector<KeyPoint>& kpts_query, vector<Point2f>& pts_train, vector<Point2f>& pts_query,
                    vector<int> &goodkpts_1)
{
  pts_train.clear();
  pts_query.clear();
  pts_train.reserve(matches.size());
  pts_query.reserve(matches.size());
  for (size_t i = 0; i < matches.size(); i++)
  {
    const DMatch& match = matches[i];
    pts_query.push_back(kpts_query[match.queryIdx].pt);
    pts_train.push_back(kpts_train[match.trainIdx].pt);
    goodkpts_1.push_back(match.trainIdx);
  }

}

static double match(const vector<KeyPoint>& /*kpts_train*/, const vector<KeyPoint>& /*kpts_query*/, Ptr<DescriptorMatcher> matcher,
            const Mat& train, const Mat& query, vector<DMatch>& matches)
{

  double t = (double)getTickCount();
  matcher->match(query, train, matches); //Using features2d
  return ((double)getTickCount() - t) / getTickFrequency();
}


void SVO::stereo_match( string im1_name, string im2_name, Mat &Pl, Mat &Pr, Mat &Kl, Mat &Kr, Mat &Dl, Mat &Dr, Mat &rvec, Mat &tvec, Mat &outimg)
{
  Mat im1= imread(im1_name, CV_LOAD_IMAGE_GRAYSCALE);
  Mat im2 = imread(im2_name, CV_LOAD_IMAGE_GRAYSCALE);

  //Mat im1, im2;
  //undistort(im1dist, im1, Kl, Mat() /*Dl*/);
  //undistort(im2dist, im2, Kr, Mat() /*Dr*/);

  //Rectification(im1, im2, Kl, Kr, Dl, Dr, tvec);

  if (im1.empty() || im2.empty())
  {
    cout << "could not open one of the images..." << endl;
    cout << "the cmd parameters have next current value: " << endl;
    exit -1;
  }

  vector<KeyPoint> kpts_1, kpts_2;

  SIFT_detector->detect(im1, kpts_1);
  SIFT_detector->detect(im2, kpts_2);


  //cout << "found " << kpts_1.size() << " keypoints in " << im1_name << endl << "fount " << kpts_2.size()
     // << " keypoints in " << im2_name << endl << "took " << t << " seconds." << endl;

  Mat desc_1, desc_2;

  //cout << "computing descriptors..." << endl;
  
  SIFT_descriptor->compute(im1, kpts_1, desc_1);
  SIFT_descriptor->compute(im2, kpts_2, desc_2);

  //cout << "desc_1.size: " << desc_1.rows << " , " << desc_1.cols << endl;
  //cout << "desc_2.size: " << desc_2.rows << " , " << desc_2.cols << endl;

  if ( desc_1.empty() )
   cvError(0,"MatchFinder","1st descriptor empty",__FILE__,__LINE__);    
  if ( desc_2.empty() )
   cvError(0,"MatchFinder","2nd descriptor empty",__FILE__,__LINE__);

  //Do matching using features2d
  //cout << "matching with BruteForceMatcher<Hamming>" << endl;
  //BFMatcher matcher_popcount();
  
  vector<DMatch> matches;
  double pop_time = match(kpts_1, kpts_2, SIFT_matcher, desc_1, desc_2, matches);
  //cout << "done BruteForceMatcher<Hamming> matching. took " << pop_time << " seconds" << endl;
  
  //remove bad matches
  removeMatches(matches);
  
  vector<Point2f> mpts_1, mpts_2;
  vector<int> goodkpts_1;

  matches2points(matches, kpts_1, kpts_2, mpts_1, mpts_2, goodkpts_1); //Extract a list of the (x,y) location of the matches
  //cout << "matches converted to points" << endl;
  vector<unsigned char> outlier_mask;
  //cout << mpts_2.size() << " , " << mpts_1.size() << endl;
  Mat H = findHomography(mpts_2, mpts_1, RANSAC, 1, outlier_mask);
  //cout << "homography computed" << endl;

  vector<char> mask(outlier_mask.size(), 1) ;
  int j =0;
  for (int i= 0; i< int(outlier_mask.size()); ++i)
  {
    //cout << "entered for loop" << endl;
       mask[i] = char(outlier_mask[i]);
       if (int(outlier_mask[i]) == 0)
       {
          mpts_1.erase(mpts_1.begin() + j);
          mpts_2.erase(mpts_2.begin() + j);
          goodkpts_1.erase(goodkpts_1.begin() + j);
          j--;
       }
      j++;
        
  }
  //traingulate
  Mat currpt_3d_h(4, mpts_1.size(),CV_32FC1);
  cv::triangulatePoints(Pl,Pr,mpts_1,mpts_2,currpt_3d_h);
  Mat currpt_3d; convertPointsFromHomogeneous(Mat(currpt_3d_h.t()).reshape(4, 1),currpt_3d);
  //   cout << currpt_3d.size() << endl;
  //    cout << currpt_3d.rowRange(0,10) << endl;
  vector<VectorXd> features3d;
  vector<uchar> status(currpt_3d.rows,0);
  for (int i=0; i<currpt_3d.rows; i++) {
      VectorXd feature3d(6) ;
      feature3d << goodkpts_1[i], mpts_1[i].x, mpts_1[i].y, currpt_3d.at<Point3f>(i).x, currpt_3d.at<Point3f>(i).y, currpt_3d.at<Point3f>(i).z;
      
      //cout << mpts_1[i] << endl;
      features3d.push_back(feature3d);

      //cout << currpt_3d.at<Point3f>(i) << endl;

      status[i] = (currpt_3d.at<Point3f>(i).z > 0) ? 1 : 0;
      if(status[i] == 0)
      {
        cout << "negative world coordinate detected" << endl;
        cout << currpt_3d.at<Point3f>(i) << endl;
      }
     //cout << "3d point " << i << " : " << currpt_3d.at<Point3f>(i) << endl;
  }
  int count = countNonZero(status);
  //assert(count == goodmpts_ind.size());
  //cout << "#final world points: " << count << endl;

  //cout << "#final matches: " << H_matches << endl;

  //calculate reprojection
  
  vector<Point2f> reprojected_pt_set1;

 // projectPoints(currpt_3d,rvec,tvec,Kl,Mat(),reprojected_pt_set1);
  //cout << "after projectPoints" << endl;
  //double reprojErr = cv::norm(reprojected_pt_set1,mpts_1,NORM_L2)/(double)mpts_1.size();
  //cout << "reprojected_pt_set1 size: " << reprojected_pt_set1.size() <<
  //" , " << "mpts_1 size: " <<  mpts_1.size() << endl;
  //draw3DPoints(im1, reprojected_pt_set1);
  //cout << "reprojection Error " << reprojErr << endl;

  //Mat imgmatches;
 
  //drawMatches(im2, kpts_2, im1, kpts_1, matches, imgmatches, Scalar::all(-1), Scalar::all(-1), mask);
  //imshow("stereo match 3Dpoints", im1);
  //waitKey(1);

  //cout << "currpt3d: " << currpt_3d << endl;
  //cout << "prevpt3d: " << this->prevpt_3d << endl;
  //Mat inputimg = imread(im1_name);
  
  this->featureMatching(im1, outimg, kpts_1, desc_1, features3d, Kl, Dl);
  this->prevfeatures3d = features3d;


  
}


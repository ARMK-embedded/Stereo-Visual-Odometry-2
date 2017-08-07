
#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <ctime>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/nonfree/nonfree.hpp"

using namespace cv;
using namespace Eigen;
using namespace std;



class SVO{


public:

	SVO(); // default constructor
	~SVO(); // default destructor

	VectorXd X;
	MatrixXd P;
	MatrixXd H;
	Matrix3d Q;
	MatrixXd POSE;

	Mat Kl;
	Mat Kr;
	Mat Dl;
	Mat Dr;

	
	Mat prevImg;
	int imgCount;
	Matrix2d focal;
	Vector2d pp;
	Mat prevDescriptors;
	vector<KeyPoint> prevKeypoints;
	vector<DMatch> stereo_matches;
	vector<VectorXd> prevfeatures3d;
	vector<Point2f> prevcorrespts2D;

	cv::Mat SIFT_H_prev;
	cv::Ptr<cv::FeatureDetector> SIFT_detector;
	cv::Ptr<cv::DescriptorExtractor> SIFT_descriptor;
	cv::Ptr<cv::DescriptorMatcher> SIFT_matcher;
	std::vector<cv::KeyPoint> SIFT_train_kpts;
	std::vector<Point2f> SIFT_train_pts;
	cv::Mat SIFT_train_desc;


	//cv::Ptr<cv::FeatureDetector> detector;
	//cv::Ptr<cv::DescriptorExtractor> extractor;

	void Xinit();
	void featureInit();
	void stereo_match(string im1_name, string im2_name, Mat &Pl, Mat &Pr, Mat &Kl, Mat &Kr, Mat &Dl, Mat &Dr, Mat &rvec, Mat &tvec,
						Mat &outimg);
	void temporal_match(Mat &currImg, vector<KeyPoint> &currGoodKpts, Mat &currGoodDesc, Mat &currpt3d);

	void featureMatching(Mat &img, Mat &SIFT_outputImg, vector<KeyPoint> &SIFT_query_kpts,
                          Mat &SIFT_query_desc, vector<VectorXd> &features3d, Mat &cameraMatrix, 
                          Mat &distCoeffs);
	void correspondence3D_3D(vector<DMatch> &matches, vector<VectorXd> &features3d, Mat &img, vector<Point2f> &correspts2D);
	void removeMatches(vector<DMatch> &matches);
	void GaussNewton(vector<Vector3d> &p1, vector<Vector3d> &p2, VectorXd &XGN, MatrixXd &PGN);
	void RansacGN(vector<Vector3d> &p1, vector<Vector3d> &p2);
};




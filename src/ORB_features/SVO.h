#ifndef MSCKF_H_
#define MSCKF_H_


#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <ctime>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <sstream>
#define MAXTRACK 30
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

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

	Mat Kl(3, 3, CV_64F);
	Mat Kr(3, 3, CV_64F);
	Mat Dl(5, 1, CV_64F);
	Mat Dr(5, 1, CV_64F);

	
	Mat img;
	Matrix2d focal;
	Vector2d pp;

	// ORB
	Mat ORB_H_prev;
	vector<MatrixXd> features;
	vector<Vector2i > featuresIdx; // to be filled with current train indices
	vector<MatrixXd> lostfeatures;
    vector<int> lostfeaturesCamIdx;
 	int imageNum = 0;

	Ptr<cv::OrbFeatureDetector> ORB_detector;
	//cv::Ptr<cv::FeatureDetector> ORB_detector;
	vector<cv::KeyPoint> ORB_train_kpts;

	vector<Point2f> ORB_train_pts;

	Ptr<cv::DescriptorExtractor> ORB_descriptor;
	Mat ORB_train_desc;

	Ptr<cv::DescriptorMatcher> ORB_matcher;


	void Xinit();


    void runFeatureMatching(Mat &img, Mat &ORB_outputImg);
		void featuresInit();
		void matches2points(const vector<KeyPoint>& train, const vector<KeyPoint>& query,
			    const std::vector<cv::DMatch>& matches, std::vector<cv::Point2f>& pts_train,
			    std::vector<Point2f>& pts_query);
		void augmentFeatures(vector<KeyPoint> ORB_query_kpts);
		void augmentoldFeatures(vector<DMatch> matches, vector<KeyPoint> ORB_query_kpts);
		void featureMatching(Mat &img, Mat &ORB_outputImg);


   	struct 
   	{
   		double sigma_img;
   		double fx;
   		double fy;
		double px;
		double py;

   	} CameraParams;


};

#endif

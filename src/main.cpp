/*main file for processing the data and running the EKF*/


// #include "features.h"
// #include "EKF.h"


#define MAX_FRAME 4000
#define MIN_NUM_FEAT 2000

#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <limits>

#include "SVO.h"
#include "../fast-cpp-csv-parser/csv.h"
#include <math.h>
//Include headers for OpenCV Image processing
#include <opencv2/imgproc/imgproc.hpp>
//Include headers for OpenCV GUI handling
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>



using namespace std;
//using Eigen::VectorXd;
//using Eigen::MatrixXd;
using namespace Eigen;
using namespace cv;



int main( int argc, char** argv )
{
  Mat inputImg;
  Mat outputImg;

  ofstream output;
  output.open ("voOutput.txt");

 /* io::CSVReader<2> leftcamera_data("/home/anoop/Documents/robotics/EKF_mono_slam/mav0/cam0/data.csv"); 
  leftcamera_data.read_header(io::ignore_extra_column, "#timestamp [ns]", "filename");
  double leftcam_time; std::string leftImg;

  io::CSVReader<2> rightcamera_data("/home/anoop/Documents/robotics/EKF_mono_slam/mav0/cam1/data.csv"); 
  rightcamera_data.read_header(io::ignore_extra_column, "#timestamp [ns]", "filename");
  double rightcam_time; std::string rightImg;*/

  String leftfolder = "/home/anoop/Documents/robotics/EKF_mono_slam/KITTI01/image_2/" ;
  //"/home/anoop/Documents/robotics/EKF_mono_slam/mav0/cam0/data/*.png";
  vector<String> leftcam;
  cv::glob(leftfolder, leftcam);
  String rightfolder = "/home/anoop/Documents/robotics/EKF_mono_slam/KITTI01/image_3/";
  //"/home/anoop/Documents/robotics/EKF_mono_slam/mav0/cam1/data/*.png";
  vector<String> rightcam;
  cv::glob(rightfolder, rightcam);
  

  Mat Kl(3, 3, CV_64F);
  Kl.at<double>(0,0) = 7.188560000000e+02 /*458.654*/;  Kl.at<double>(0,1) = 0.; Kl.at<double>(0,2) = 6.071928000000e+02  /*367.215*/;
  Kl.at<double>(1,0) = 0.; Kl.at<double>(1,1) = 7.188560000000e+02 /*457.296*/; Kl.at<double>(1,2) = 1.852157000000e+02 /*248.375*/;
  Kl.at<double>(2,0) = 0.; Kl.at<double>(2,1) = 0.; Kl.at<double>(2,2) = 1.;

  Mat Kr(3, 3, CV_64F);
  Kr.at<double>(0,0) = 7.188560000000e+02 /*457.587*/;  Kr.at<double>(0,1) = 0.; Kr.at<double>(0,2) = 6.071928000000e+02 /*379.999*/;
  Kr.at<double>(1,0) = 0.; Kr.at<double>(1,1) = 7.188560000000e+02 /*456.134*/; Kr.at<double>(1,2) = 1.852157000000e+02 /*255.238*/;
  Kr.at<double>(2,0) = 0.; Kr.at<double>(2,1) = 0.; Kr.at<double>(2,2) = 1.;

  Mat Dl(4, 1, CV_64F);
  Dl.at<double>(0,0) = -0.28340811;
  Dl.at<double>(0,1) = 0.07395907;
  Dl.at<double>(0,2) = 0.00019359; 
  Dl.at<double>(0,3) = 1.76187114e-05;
  //Dl.at<double>(0,4) = 0.0;

  Mat Dr(4, 1, CV_64F);
  Dr.at<double>(0,0) = -0.28368365;
  Dr.at<double>(0,1) =  0.07451284;
  Dr.at<double>(0,2) =  -0.00010473; 
  Dr.at<double>(0,3) = -3.55590700e-05;
  //Dr.at<double>(0,4) = 0.0;

 /* Matrix4d Rtl, Rtr;
  Rtl <<  0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
         0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
         0.0, 0.0, 0.0, 1.0;
  Rtr << 0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
         0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
        -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
         0.0, 0.0, 0.0, 1.0;
  Matrix3d Rl;
  Rl << 0.0148655429818, -0.999880929698, 0.00414029679422, 
         0.999557249008, 0.0149672133247, 0.025715529948,
        -0.0257744366974, 0.00375618835797, 0.999660727178;
  Vector3d tl( -0.0216401454975,-0.064676986768,  0.00981073058949);

  MatrixXd tempC = MatrixXd::Zero(3,4);
  tempC << MatrixXd::Identity(3,3), MatrixXd::Zero(3,1);

  cout << tempC << endl;

  Matrix3d Kle, Kre;
  cv::cv2eigen(Kl,Kle);
  cv::cv2eigen(Kr,Kre);
  cout << Kle << endl;
  cout << Kre << endl;


  MatrixXd Ple, Pre;
  Ple = Kle*tempC*Rtl;
  Pre = Kre*tempC*Rtr;*/

  Mat Pl, Pr;
  Pl = Mat::zeros(3,4, CV_64F);
  Pr = Mat::zeros(3,4, CV_64F);
 // cv::eigen2cv( Ple, Pl );
 // cv::eigen2cv( Pre, Pr );
  Pl.at<double>(0,0) = 7.188560000000e+02 ;
  Pl.at<double>(1,1) = 7.188560000000e+02 ;
  Pl.at<double>(0,2) = 6.071928000000e+02;
  Pl.at<double>(1,2) = 1.852157000000e+02;
  Pl.at<double>(2,2) = 1.0;

  Pr.at<double>(0,0) = 7.188560000000e+02 ;
  Pr.at<double>(1,1) = 7.188560000000e+02 ;
  Pr.at<double>(0,2) = 6.071928000000e+02;
  Pr.at<double>(1,2) = 1.852157000000e+02;
  Pr.at<double>(2,2) = 1.0;
  Pr.at<double>(0,3) = -3.861448000000e+02;
  cout << "Pl: " << Pl << endl;
  cout << "Pr: " << Pr << endl;

  //Decomposing the projection matrix
 // cv::Mat K(3,3,cv::DataType<double>::type); // intrinsic parameter matrix
 // cv::Mat R(3,3,cv::DataType<double>::type); // rotation matrix
 // eigen2cv(Rl, R);
  //cv::Mat Thomogeneous(4,1,cv::DataType<double>::type); // translation vector
  //cv::decomposeProjectionMatrix(Pl, K, R, Thomogeneous);
  //std::cout << "K: " << K << std::endl;
  //std::cout << "R: " << R << std::endl;
  //std::cout << "Thomogeneous: " << Thomogeneous << std::endl;
  cv::Mat rvec(3,1,cv::DataType<double>::type);//rodrigues rotation matrix
  //cv::Rodrigues(R,rvec);
  //cout << "rvec: " << rvec << endl;

  cv::Mat T(3,1,cv::DataType<double>::type); // translation vector
  //eigen2cv(tl, T);
  //cv::Mat T;
  //cv::convertPointsHomogeneous(Thomogeneous, T);
  //std::cout << "T: " << T << std::endl;


  /*double left_focal = 458.654;
  cv::Point2d left_pp(367.215, 248.375);

  double right_focal = 457.587;
  cv::Point2d right_pp(379.999, 255.238);*/
  
  //namedWindow( "Camerafeed", WINDOW_AUTOSIZE );// Create a window for display.

  SVO filter;
  filter.Xinit();
  filter.featureInit();
  filter.Kl = Kl;
  filter.Kr = Kr;
  filter.Dl = Dl;
  filter.Dr = Dr;


  /*Quaterniond orient = Quaterniond( 0.534108, -0.153029,  -0.827383,  -0.082152);

  cout << orient.coeffs().transpose() << endl;
  
  
  double yaw, pitch, roll;
  Vector3d euler = orient.toRotationMatrix().eulerAngles(2, 1, 0);
  //yaw : Z , pitch: Y, roll: X
  yaw = euler[0]; pitch = euler[1]; roll = euler[2]; 
  //initialise the 6DOF of the uav
 // filter.X << yaw, pitch, roll,Vector3d(4.688319,  -1.786938,  0.783338); 
  filter.X(0) = yaw;
  filter.X(1) = pitch;
  filter.X(2) = roll;
  filter.X.segment<3>(3) = Vector3d(4.688319,  -1.786938,  0.783338);*/


  int framenum = 0;

  filter.stereo_match(leftcam[framenum], rightcam[framenum], Pl, Pr, Kl, Kr, Dl, Dr, rvec, T, outputImg);
  framenum++;
  while( framenum <= 1100  /*(leftcamera_data.read_row(leftcam_time, leftImg)) && (rightcamera_data.read_row(rightcam_time, rightImg))*/)
  {
      cout << "Image#: " <<  framenum << endl;
      filter.imgCount++;

      //imgl = imread(leftcam[framenum]);

     /* if(! imgl.data || imgl.empty() )  {                            // Check for invalid input
          cout <<  "Could not open or find the image" << std::endl ;
          return -1;  //break;
      }*/

      if (true   /*fabs(leftcam_time- rightcam_time)/1e+9  < 1e-3*/)
      {

        filter.stereo_match(leftcam[framenum], rightcam[framenum], Pl, Pr, Kl, Kr, Dl, Dr, rvec, T, outputImg);
        inputImg = cv::imread(leftcam[framenum]);
        //filter.featureMatching(inputImg, outputImg);
      }

      /*filter.runFeatureMatching(filter.img, outputImg);
      cout << "covariance size: " <<  filter.covariance.rows() << ',' << filter.covariance.cols() << endl;
      filter.augmentFilter();
      filter.stackingResidualsFeature();
      
      cout << "imageNum: " << filter.imageNum << endl;

      imshow("Camerafeed", outputImg);
      waitKey(1);  */ 

      Vector3d pose = filter.POSE.block<3,1>(0,3);
      cout << pose.transpose() << endl;
      output << (pose(0)) << ' ' << (pose(1)) << ' ' << (pose(2)) << endl;  
           
  framenum++; // increment the image counter

  }                                  
 

 return 0;

}



#include "SVO.h"
using namespace cv;
using namespace Eigen;
using namespace std;

void SVO::featureInit()
{
  // SIFT
  SIFT_detector = new cv::SiftFeatureDetector();
  SIFT_descriptor = new cv::SiftDescriptorExtractor();
  SIFT_matcher = DescriptorMatcher::create("BruteForce");

}



void drawMatchesRelative(const vector<KeyPoint>& train, const vector<KeyPoint>& query,
        std::vector<cv::DMatch>& matches, Mat& img, const vector<unsigned char>& mask = vector<
        unsigned char> ())
    {
        int matchesCounter = 0;
        for (int i = 0; i < (int)matches.size(); i++)
        {
            if (mask.empty() || mask[i])
            {
                matchesCounter++;
                Point2f pt_new = query[matches[i].queryIdx].pt;
                Point2f pt_old = train[matches[i].trainIdx].pt;

                cv::line(img, pt_new, pt_old, Scalar(0, 0, 255), 2);
                cv::circle(img, pt_new, 2, Scalar(255, 0, 0), 1);

            }
        }
        cout << "matchesCounter: " << matchesCounter << endl;
    }

void drawCorrespondence( Mat& img, vector<Point2f> &pts_new, vector<Point2f> &pts_old)
    {

        for (int i = 0; i < (int)pts_new.size(); i++)
        {
            
                Point2f pt_new = pts_new[i];
                Point2f pt_old = pts_old[i];

                //cv::line(img, pt_new, pt_old, Scalar(0, 0, 255), 2);
                cv::circle(img, pt_new, 2, Scalar(255, 0, 0), 1);
                cv::circle(img, pt_old, 2, Scalar(0, 0, 255), 1);
        }
    }

//Takes a descriptor and turns it into an xy point
void keypoints2points(const vector<KeyPoint>& in, vector<Point2f>& out)
{
    out.clear();
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i)
    {
        out.push_back(in[i].pt);
    }
}

//Takes an xy point and appends that to a keypoint structure
void points2keypoints(const vector<Point2f>& in, vector<KeyPoint>& out)
{
    out.clear();
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i)
    {
        out.push_back(KeyPoint(in[i], 1));
    }
}

//Uses computed homography H to warp original input points to new planar position
void warpKeypoints(const Mat& H, const vector<KeyPoint>& in, vector<KeyPoint>& out)
{
    vector<Point2f> pts;
    keypoints2points(in, pts);
    vector<Point2f> pts_w(pts.size());
    Mat m_pts_w(pts_w);
    perspectiveTransform(Mat(pts), m_pts_w, H);
    points2keypoints(pts_w, out);
}

//Converts matching indices to xy points
void matches2points(const vector<KeyPoint>& train, const vector<KeyPoint>& query,
    const std::vector<cv::DMatch>& matches, std::vector<cv::Point2f>& pts_train,
    std::vector<Point2f>& pts_query)
{

    pts_train.clear();
    pts_query.clear();
    pts_train.reserve(matches.size());
    pts_query.reserve(matches.size());

    size_t i = 0;

    for (; i < matches.size(); i++)
    {

        const DMatch & dmatch = matches[i];

        pts_query.push_back(query[dmatch.queryIdx].pt);
        pts_train.push_back(train[dmatch.trainIdx].pt);
       // cout << dmatch.trainIdx << endl;

    }

}


void resetH(Mat&H)
{
    H = Mat::eye(3, 3, CV_32FC1);
}




//Note Prev => Train, Curr => Query
void SVO::featureMatching(Mat &img, Mat &SIFT_outputImg, vector<KeyPoint> &SIFT_query_kpts,
                          Mat &SIFT_query_desc, vector<VectorXd> &features3d, Mat &cameraMatrix, 
                          Mat &distCoeffs)
{
  cv::Mat imGraydistort;
  if(img.channels() == 3)
      cvtColor(img, imGraydistort, CV_RGB2GRAY);
  else
      img.copyTo(imGraydistort);
  
  Mat imGray;
  undistort(imGraydistort, imGray, cameraMatrix, distCoeffs);
  std::vector<Point2f>  SIFT_query_pts;
  std::vector<cv::DMatch> SIFT_matches;
  std::vector<unsigned char> SIFT_match_mask;
  vector<Point2f> correspts2D;

  // SIFT...
  img.copyTo(SIFT_outputImg);
  cout << "SIFT_train_kpts is empty ?: " << SIFT_train_kpts.empty() << endl;
  if(SIFT_H_prev.empty())
    SIFT_H_prev = Mat::eye(3,3,CV_32FC1);

  if(!SIFT_train_kpts.empty())
  {

    std::vector<cv::KeyPoint> test_kpts;
    warpKeypoints(SIFT_H_prev.inv(), SIFT_query_kpts, test_kpts);
    cv::Mat SIFT_mask = windowedMatchingMask(test_kpts, this->SIFT_train_kpts, 25, 25);
    SIFT_matcher->match(SIFT_query_desc, this->SIFT_train_desc, SIFT_matches, SIFT_mask);
    removeMatches(SIFT_matches);// remove redundant matches
    matches2points(this->SIFT_train_kpts, SIFT_query_kpts, SIFT_matches, SIFT_train_pts, SIFT_query_pts);

    
    
    if(SIFT_matches.size() > 5)
    {
      cv::Mat H = findHomography(SIFT_train_pts, SIFT_query_pts, RANSAC, 4, SIFT_match_mask);
     /* int j= 0;
      for(size_t i= 0; i< SIFT_matches.size(); ++i)
      {
        if(int(SIFT_match_mask[i]) == 0)
        {
          SIFT_matches.erase(SIFT_matches.begin() + j);
          --j;

        }
        ++j;

      }*/

      if(countNonZero(Mat(SIFT_match_mask)) > 10)
      {
         SIFT_H_prev = H;
        //correspondence3D_3D(SIFT_matches, features3d, SIFT_outputImg, correspts2D);
        cout << "temporal matches count: " << countNonZero(Mat(SIFT_match_mask)) << endl;
      }
      else
        SIFT_H_prev = Mat::eye(3,3,CV_32FC1);

      drawMatchesRelative(SIFT_train_kpts, SIFT_query_kpts, SIFT_matches, SIFT_outputImg, SIFT_match_mask);
      //process prevfeatures3d
      //prevfeatures3d has train indices and features3d has query indices
      //matches has match.trainIdx, match.queryIdx
      vector<Vector3d> curr3d, prev3d;
      vector<Point2f> curr2d, prev2d;
        for (auto match: SIFT_matches)                    //for all i in matches
      {
        // cout << "match indices" << match.queryIdx << " , " << match.trainIdx << endl;          //search in features3d indices for the query  index
          for (int j=0; j< int(features3d.size()); ++j) {        //for all j in features3d
            if (match.queryIdx == features3d[j](0))
            {
              //save the 3d point
              curr3d.push_back(features3d[j].segment<3>(3));
              curr2d.push_back(Point2f(features3d[j](1),features3d[j](2)));
              break;
            }
          }
              
          //search in prevfeatures3d indices for the train index
          for (int j=0; j< int(prevfeatures3d.size()); ++j)   {   //for all j in prevfeatures3d
            if (match.trainIdx == prevfeatures3d[j](0)){
              //save the 3dpoint
              prev3d.push_back(prevfeatures3d[j].segment<3>(3));
              prev2d.push_back(Point2f(prevfeatures3d[j](1),prevfeatures3d[j](2)));
              break;
            }
              
          }

      } // correspondence for loop
      //prev2d.resize(curr2d.size());
      prev3d.resize(curr3d.size());
      drawCorrespondence(SIFT_outputImg, curr2d, prev2d);
      imshow("3Dcorrespondce", SIFT_outputImg);
      waitKey(1);  

      cout << "3d correspondence size: " << curr3d.size() << " , " << prev3d.size() << endl;
      if( curr3d.size()> 50)//do gauss newton only if points are > 50
        RansacGN(prev3d, curr3d);
    } // if matches > 5
  } //if !trainkpts.empty
  else
  { 
    SIFT_H_prev = Mat::eye(3,3,CV_32FC1);
  }

  this->SIFT_train_kpts = SIFT_query_kpts;
  SIFT_query_desc.copyTo(this->SIFT_train_desc); 
  
  if(true)
    cout << ", SIFT temporal matches: " << SIFT_matches.size() << endl;

}

/*void SVO::correspondence3D_3D(vector <DMatch> &matches, vector<VectorXd> &features3d, Mat &SIFT_outputImg,
                              vector<Point2f> &correspts2D)
{
  
  cout << "matches size: " << matches.size() << endl;
  cout << "features3d size" << features3d.size() << endl;
  int count = 0;
 // for (size_t i=0; i< matches.size(); ++i)
  //{
    for(size_t j=0; j< features3d.size(); ++j)
    {
      //if(matches[i].queryIdx == features3d[j](0)){  //matches[i].trainIdx == prevfeatures3d[j](0)
        //cout << matches[i].queryIdx << " , " << matches[i].trainIdx << endl;
        correspts2D.push_back(Point2f(features3d[j](1), features3d[j](2)));
        //count++;
        //break;
      //}

    }

    for(size_t j=0; j< this->prevfeatures3d.size(); ++j)
    {
      //if(matches[i].trainIdx == this->prevfeatures3d[j](0)){  //matches[i].trainIdx == prevfeatures3d[j](0)
        //cout << matches[i].queryIdx << " , " << matches[i].trainIdx << endl;
        this->prevcorrespts2D.push_back(Point2f(this->prevfeatures3d[j](1), this->prevfeatures3d[j](2)));
        //count++;
        //break;
      //}

    }


  //}
  cout << correspts2D.size() << " , " << prevcorrespts2D.size() << endl;
  drawCorrespondence(SIFT_outputImg, correspts2D, prevcorrespts2D);
  //imshow("Camerafeed", SIFT_outputImg);
  //waitKey(1); 
  //cout << "correspondence3D_3D counter: " << count << endl;
  

}*/
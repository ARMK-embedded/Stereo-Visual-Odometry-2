#include "MSCKF.h"
using namespace cv;
using namespace Eigen;
using namespace std;


////////////////////////////////////////////////////////////////////////////////////////////////
//feature matching module
void SVO::featuresInit(VectorXd &X, MatrixXd &P)
{

    this->X = X;
    this->P = P;
    this->H = MatrixXd::Zero(3,6);
    this->Q = 0.001*Matrix3d::Identity();

    focal  << CameraParams.fx, 0,
             0, CameraParams.fy;
    pp = Vector2d(CameraParams.px, CameraParams.py); // principal point 

    vector<MatrixXd> features;
    vector<Vector2i> featuresIdx;
    vector<MatrixXd> lostfeatures;
    vector<int> lostfeaturesCamIdx;
    ORB_detector = new cv::OrbFeatureDetector();
    ORB_descriptor = new cv::OrbDescriptorExtractor();
    ORB_matcher = DescriptorMatcher::create("BruteForce-Hamming");


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
void SVO::matches2points(const vector<KeyPoint>& train, const vector<KeyPoint>& query,
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

        //augmentoldFeatures( pts_query[i], dmatch.trainIdx, dmatch.queryIdx);
       // cout << "trainIdx: " << dmatch.trainIdx << endl ;

    }

}



void resetH(Mat&H)
{
    H = Mat::eye(3, 3, CV_32FC1);
}



//Note Prev => Train, Curr => Query
void SVO::featureMatching(Mat &img, Mat &ORB_outputImg)
{

            // ORB
    
    vector<cv::KeyPoint> ORB_query_kpts;
    vector<Point2f> ORB_query_pts;
    Mat  ORB_query_desc;

    vector<cv::DMatch> ORB_matches;


    cv::Mat imGray;
    if(img.channels() == 3)
        cvtColor(img, imGray, CV_RGB2GRAY);
    else
        img.copyTo(imGray);

    //cout << "cvtColor done" << endl;

    // ORB...
    img.copyTo(ORB_outputImg);
    ORB_detector->detect(imGray, ORB_query_kpts);
    ORB_descriptor->compute(imGray, ORB_query_kpts, ORB_query_desc);
    if(ORB_H_prev.empty())
        ORB_H_prev = Mat::eye(3,3,CV_32FC1);

    std::vector<unsigned char> ORB_match_mask;

    //cout << "feature detection done" << endl;
    if(!ORB_train_kpts.empty())
    {
        std::vector<cv::KeyPoint> test_kpts;
        warpKeypoints(ORB_H_prev.inv(), ORB_query_kpts, test_kpts);
        cv::Mat ORB_mask = windowedMatchingMask(test_kpts, ORB_train_kpts, 25, 25);
        ORB_matcher->match(ORB_query_desc, ORB_train_desc, ORB_matches, ORB_mask);
        
        matches2points(ORB_train_kpts, ORB_query_kpts, ORB_matches, ORB_train_pts, ORB_query_pts);
        
        
        if(ORB_matches.size() > 5)
        {
            cv::Mat H = findHomography(ORB_train_pts, ORB_query_pts, RANSAC, 4, ORB_match_mask);
            if(countNonZero(Mat(ORB_match_mask)) > 15)
                ORB_H_prev = H;
            else
                ORB_H_prev = Mat::eye(3,3,CV_32FC1);

            drawMatchesRelative(ORB_train_kpts, ORB_query_kpts, ORB_matches, ORB_outputImg, ORB_match_mask);

            //augmentoldFeatures(ORB_matches, ORB_query_kpts);
        }
    }
    else
    {   
        ORB_H_prev = Mat::eye(3,3,CV_32FC1);
        ORB_train_kpts = ORB_query_kpts;
        ORB_query_desc.copyTo(ORB_train_desc);  

        //augmentFeatures( ORB_query_kpts);

    }

    //ORB_train_kpts = ORB_query_kpts;
    //ORB_query_desc.copyTo(ORB_train_desc);  
    
    if(true)
        cout << ", ORB matches: " << ORB_matches.size() << endl;

}

void SVO::augmentoldFeatures(vector<DMatch> matches, vector<KeyPoint> ORB_query_kpts)
{

    //cout  << "entered augmentoldFeatures" << endl;

    //vector<MatrixXd> lostfeatures;
    //vector<int> lostfeaturesCamIdx;
    
    //cout << "features.size() before : " << features.size() << endl;

    int notFound =  0;
    for(int j=0; j< featuresIdx.size(); ++j)
    {  
       // cout << "entered loop j: " << j << endl;
        bool found = false;
        for (int i=0; i < matches.size() ; ++i)
        {
            //cout << "entered loop i: " << i << endl;
             DMatch dmatch = matches[i];
            //cout << "featuresIdx[j] " << featuresIdx[j][1] << endl;
            if (dmatch.trainIdx == featuresIdx[j](0)) // j
            {
                //cout << "match found" << endl;
                found = true;
                int M = features[j].rows();
                int N = features[j].cols();
                //cout << M << ',' << N << endl;
                features[j].conservativeResize(M, N+1);
                features[j](0,N) = ORB_query_kpts[dmatch.queryIdx].pt.x;
                features[j](1,N) = ORB_query_kpts[dmatch.queryIdx].pt.y;
                //featuresIdx[j](0) = dmatch.queryIdx;
                //featuresIdx[j](1) = imageNum;

                break;

            }
        }

         if(!found)
        {
            notFound++;
           // cout << j << " not found" << endl;
           // cout << "features [" << j << "] measurements " << features[j].cols() << endl;

            //cout << featuresIdx[j][1] << "-" << imageNum-1 << endl;

            
           //featuresIdx[j][0] = -1;
           lostfeatures.push_back(features[j]);
           lostfeaturesCamIdx.push_back(featuresIdx[j](1));
           cout << "no.of frames tracked: " << featuresIdx[j](1) << '-' << imageNum-1 << endl;
           //cout << "j: " << j << endl;
           features.erase(features.begin() + j);
           featuresIdx.erase(featuresIdx.begin() + j);
            j = j - 1;
          

        }
        //cout << "features.size() after: " << features.size() << endl;

    }

    cout << "lostfeatures.size: " << lostfeatures.size() << endl;
    if (notFound > 400)
            cout << "notFound counter " << notFound << " running low on features" << endl;

}


void SVO::augmentFeatures(vector<KeyPoint> ORB_query_kpts)
{
    //cout << "entered augmentFeatures" << endl;
    features = vector<MatrixXd> (ORB_query_kpts.size(), MatrixXd::Zero(2,1));
    featuresIdx = vector<Vector2i> (ORB_query_kpts.size(), Vector2i::Zero());
    //cout << featuresIdx[0][0] << endl;
    for(int i= 0; i< int(ORB_query_kpts.size()) ; ++i)
    {
        features[i](0,0) = ORB_query_kpts[i].pt.x;
        features[i](1,0) = ORB_query_kpts[i].pt.y;
        featuresIdx[i] = Vector2i(i,imageNum);
    }

}


void SVO::runFeatureMatching(Mat &inputImg, Mat &outputImg)
{
    //cout << "ran featureMatching" << endl;
    // ORB...
    this->imageNum++;

    if (features.size() < 50)
    {
        ORB_train_kpts.clear();
        ORB_train_desc.release();
        features.clear();
        featuresIdx.clear();
        marginalizefilter();
    }
    
    lostfeatures.clear();
    lostfeaturesCamIdx.clear();
    
    featureMatching(inputImg, outputImg);

    return ;

}


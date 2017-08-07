

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <stdio.h>

using namespace cv;
using namespace Eigen;



static void print_help()
{
    printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");
    printf("\nUsage: stereo_match <left_image> <right_image> [--algorithm=bm|sgbm|hh|var] [--blocksize=<block_size>]\n"
           "[--max-disparity=<max_disparity>] [--scale=scale_factor>]\n"
           "[--no-display] [-o <disparity_image>] [-p <point_cloud_file>]\n");
}

static void saveXYZ(const char* filename, const Mat& mat)
{
    const double max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");
    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y, x);
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
            fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
        }
    }
    fclose(fp);
}

int main(int argc, char** argv)
{
    const char* algorithm_opt = "--algorithm=";
    const char* maxdisp_opt = "--max-disparity=";
    const char* blocksize_opt = "--blocksize=";
    const char* nodisplay_opt = "--no-display";
    const char* scale_opt = "--scale=";

    if(argc < 3)
    {
        print_help();
        return 0;
    }
    const char* img1_filename = 0;
    const char* img2_filename = 0;
    const char* disparity_filename = 0;
    const char* point_cloud_filename = 0;

    enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2, STEREO_VAR=3 };
    int alg = STEREO_SGBM;
    int SADWindowSize = 0, numberOfDisparities = 0;
    bool no_display = false;
    float scale = 1.f;

    StereoBM bm;
    StereoSGBM sgbm;
    StereoVar var;

    for( int i = 1; i < argc; i++ )
    {
        if( argv[i][0] != '-' )
        {
            if( !img1_filename )
                img1_filename = argv[i];
            else
                img2_filename = argv[i];
        }
        else if( strncmp(argv[i], algorithm_opt, strlen(algorithm_opt)) == 0 )
        {
            char* _alg = argv[i] + strlen(algorithm_opt);
            alg = strcmp(_alg, "bm") == 0 ? STEREO_BM :
                  strcmp(_alg, "sgbm") == 0 ? STEREO_SGBM :
                  strcmp(_alg, "hh") == 0 ? STEREO_HH :
                  strcmp(_alg, "var") == 0 ? STEREO_VAR : -1;
            if( alg < 0 )
            {
                printf("Command-line parameter error: Unknown stereo algorithm\n\n");
                print_help();
                return -1;
            }
        }
        else if( strncmp(argv[i], maxdisp_opt, strlen(maxdisp_opt)) == 0 )
        {
            if( sscanf( argv[i] + strlen(maxdisp_opt), "%d", &numberOfDisparities ) != 1 ||
                numberOfDisparities < 1 || numberOfDisparities % 16 != 0 )
            {
                printf("Command-line parameter error: The max disparity (--maxdisparity=<...>) must be a positive integer divisible by 16\n");
                print_help();
                return -1;
            }
        }
        else if( strncmp(argv[i], blocksize_opt, strlen(blocksize_opt)) == 0 )
        {
            if( sscanf( argv[i] + strlen(blocksize_opt), "%d", &SADWindowSize ) != 1 ||
                SADWindowSize < 1 || SADWindowSize % 2 != 1 )
            {
                printf("Command-line parameter error: The block size (--blocksize=<...>) must be a positive odd number\n");
                return -1;
            }
        }
        else if( strncmp(argv[i], scale_opt, strlen(scale_opt)) == 0 )
        {
            if( sscanf( argv[i] + strlen(scale_opt), "%f", &scale ) != 1 || scale < 0 )
            {
                printf("Command-line parameter error: The scale factor (--scale=<...>) must be a positive floating-point number\n");
                return -1;
            }
        }
        else if( strcmp(argv[i], nodisplay_opt) == 0 )
            no_display = true;
        else if( strcmp(argv[i], "-o" ) == 0 )
            disparity_filename = argv[++i];
        else if( strcmp(argv[i], "-p" ) == 0 )
            point_cloud_filename = argv[++i];
        else
        {
            printf("Command-line parameter error: unknown option %s\n", argv[i]);
            return -1;
        }
    }

    if( !img1_filename || !img2_filename )
    {
        printf("Command-line parameter error: both left and right images must be specified\n");
        return -1;
    }


    int color_mode = alg == STEREO_BM ? 0 : -1;
    Mat img1 = imread(img1_filename, color_mode);
    Mat img2 = imread(img2_filename, color_mode);

    if( scale != 1.f )
    {
        Mat temp1, temp2;
        int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
        resize(img1, temp1, Size(), scale, scale, method);
        img1 = temp1;
        resize(img2, temp2, Size(), scale, scale, method);
        img2 = temp2;
    }

    Size img_size = img1.size();

    Rect roi1, roi2;
    Mat Q;


    point_cloud_filename = "pointcloud.txt";

    if( point_cloud_filename  )
    {
        printf("entered point_cloud_filename if \n");

        Matrix3d Rl;
        Rl << 0.0148655429818, -0.999880929698, 0.00414029679422, 
         0.999557249008, 0.0149672133247, 0.025715529948,
        -0.0257744366974, 0.00375618835797, 0.999660727178;
        Vector3d tl( -0.0216401454975,-0.064676986768,  0.00981073058949);

        cv::Mat R(3,3,cv::DataType<double>::type); // rotation matrix
        R = Mat::eye(3,3,cv::DataType<double>::type);
        //eigen2cv(Rl, R);

        cv::Mat T(3,1,cv::DataType<double>::type); // translation vector
        eigen2cv(tl, T);
    

        Mat M1(3, 3, CV_64F), D1(5, 1, CV_64F), M2(3, 3, CV_64F), D2(5, 1, CV_64F);
        M1.at<double>(0,0) = 458.654;  M1.at<double>(0,1) = 0.; M1.at<double>(0,2) = 367.215;
        M1.at<double>(1,0) = 0.; M1.at<double>(1,1) = 457.296; M1.at<double>(1,2) = 248.375;
        M1.at<double>(2,0) = 0.; M1.at<double>(2,1) = 0.; M1.at<double>(2,2) = 1.;

        M2.at<double>(0,0) = 457.587;  M2.at<double>(0,1) = 0.; M2.at<double>(0,2) = 379.999;
        M2.at<double>(1,0) = 0.; M2.at<double>(1,1) = 456.134; M2.at<double>(1,2) = 255.238;
        M2.at<double>(2,0) = 0.; M2.at<double>(2,1) = 0.; M2.at<double>(2,2) = 1.;
        
        D1.at<double>(0,0) = -0.28340811;
        D1.at<double>(0,1) = 0.07395907;
        D1.at<double>(0,2) = 0.0;
        D1.at<double>(0,3) = 0.0;
        D1.at<double>(0,4) = 0.0;

          D2.at<double>(0,0) = -0.28368365;
          D2.at<double>(0,1) =  0.07451284;
          D2.at<double>(0,2) = 0.0;
          D2.at<double>(0,3) = 0.0;
          D2.at<double>(0,4) = 0.0;


          printf("after declaration of M and D s\n");

        M1 *= scale;
        M2 *= scale;

        Mat R1, P1, R2, P2;
        
        printf("before stereoRectify\n");

        stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );

        Mat map11, map12, map21, map22;
        initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
        initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

        Mat img1r, img2r;
        remap(img1, img1r, map11, map12, INTER_LINEAR);
        remap(img2, img2r, map21, map22, INTER_LINEAR);

       // img1 = img1r;
       // img2 = img2r;
    }

    numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width/8) + 15) & -16;

    bm.state->roi1 = roi1;
    bm.state->roi2 = roi2;
    bm.state->preFilterCap = 31;
    bm.state->SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 9;
    bm.state->minDisparity = 0;
    bm.state->numberOfDisparities = numberOfDisparities;
    bm.state->textureThreshold = 10;
    bm.state->uniquenessRatio = 15;
    bm.state->speckleWindowSize = 100;
    bm.state->speckleRange = 32;
    bm.state->disp12MaxDiff = 1;

    sgbm.preFilterCap = 63;
    sgbm.SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 3;

    int cn = img1.channels();

    sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.minDisparity = 0;
    sgbm.numberOfDisparities = numberOfDisparities;
    sgbm.uniquenessRatio = 10;
    sgbm.speckleWindowSize = bm.state->speckleWindowSize;
    sgbm.speckleRange = bm.state->speckleRange;
    sgbm.disp12MaxDiff = 1;
    sgbm.fullDP = alg == STEREO_HH;

    var.levels = 3;                                 // ignored with USE_AUTO_PARAMS
    var.pyrScale = 0.5;                             // ignored with USE_AUTO_PARAMS
    var.nIt = 25;
    var.minDisp = -numberOfDisparities;
    var.maxDisp = 0;
    var.poly_n = 3;
    var.poly_sigma = 0.0;
    var.fi = 15.0f;
    var.lambda = 0.03f;
    var.penalization = var.PENALIZATION_TICHONOV;   // ignored with USE_AUTO_PARAMS
    var.cycle = var.CYCLE_V;                        // ignored with USE_AUTO_PARAMS
    var.flags = var.USE_SMART_ID | var.USE_AUTO_PARAMS | var.USE_INITIAL_DISPARITY | var.USE_MEDIAN_FILTERING ;

    Mat disp, disp8;
    //Mat img1p, img2p, dispp;
    //copyMakeBorder(img1, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
    //copyMakeBorder(img2, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);

    int64 t = getTickCount();
    if( alg == STEREO_BM )
        bm(img1, img2, disp);
    else if( alg == STEREO_VAR ) {
        var(img1, img2, disp);
    }
    else if( alg == STEREO_SGBM || alg == STEREO_HH )
        sgbm(img1, img2, disp);
    t = getTickCount() - t;
    printf("Time elapsed: %fms\n", t*1000/getTickFrequency());

    //disp = dispp.colRange(numberOfDisparities, img1p.cols);
    if( alg != STEREO_VAR )
        disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
    else
        disp.convertTo(disp8, CV_8U);
    if( !no_display )
    {
        namedWindow("left", 1);
        imshow("left", img1);
        namedWindow("right", 1);
        imshow("right", img2);
        namedWindow("disparity", 0);
        imshow("disparity", disp8);
        printf("press any key to continue...");
        fflush(stdout);
        waitKey();
        printf("\n");
    }

    if(disparity_filename)
        imwrite(disparity_filename, disp8);

    if(point_cloud_filename)
    {
        printf("storing the point cloud...");
        fflush(stdout);
        Mat xyz;
        reprojectImageTo3D(disp, xyz, Q, true);
        saveXYZ(point_cloud_filename, xyz);
        printf("\n");
    }

    return 0;
}

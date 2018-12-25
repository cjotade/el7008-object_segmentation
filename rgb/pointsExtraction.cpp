#define _DEBUG

// Instruciones:
// Dependiendo de la versión de opencv, pueden cambiar los archivos .hpp a usar

#include <opencv2/opencv.hpp>

#include "opencv2/core/version.hpp"
#include <iostream>
#include <numeric>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;

Mat harrisFilter(Mat input)
{
	//Color to grayscale
	Mat input_gray = Mat::zeros(input.rows, input.cols, CV_32FC1);
	cvtColor(input, input_gray, CV_BGR2GRAY);

	//	1) Blur on grayscale
	Mat input_gray_blur = Mat::zeros(input.rows, input.cols, CV_32FC1);
	GaussianBlur(input_gray, input_gray_blur,Size(3,3), 0, 0);

	//	2) Derivatives in x and y directions
	Mat ix, iy;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_32FC1;
	Sobel(input_gray_blur, ix, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	Sobel(input_gray_blur, iy, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );

	//	3) Moments ixx, ixy and iyy
	Mat ixx, ixy, iyy;
	ixx = ix.mul(ix);
	ixy = ix.mul(iy);
	iyy = iy.mul(iy);

	//	4) Blur on moments
	Mat ixx_blur,ixy_blur,iyy_blur;
	GaussianBlur(ixx, ixx_blur,Size(3,3), 0, 0);
	GaussianBlur(ixy, ixy_blur,Size(3,3), 0, 0);
	GaussianBlur(iyy, iyy_blur,Size(3,3), 0, 0);

	//	5) Calc Harris
	Mat harris;
	Mat det, Tr;
	det = ixx_blur.mul(iyy_blur) - ixy_blur.mul(ixy_blur);
	Tr = ixx_blur + iyy_blur;
	harris = det - 0.04*Tr.mul(Tr);
	//cv::threshold(harris, harris, 1000000, 0, CV_THRESH_TOZERO);

	//	6) Normalize Harris
	Mat normalizedHarris;
	normalize(harris, normalizedHarris, 255, 0,NORM_MINMAX);
	Mat output;
	normalizedHarris.convertTo(output, CV_8UC1);
	return output;
}

vector<KeyPoint> getHarrisPoints(Mat harris, int val)
{
	vector<KeyPoint> points;
	double min, max;
	Point min_loc, max_loc;
	Mat mask;
	int mask_size = 5;
	//for (int i=0 ; i<harris.cols-(mask_size-1) ; i+=mask_size) {
	for (int i=0 ; i<harris.cols-(mask_size-1) ; i++) {
        //for (int j=0 ; j<harris.rows-(mask_size-1) ; j+=mask_size) {
		for (int j=0 ; j<harris.rows-(mask_size-1) ; j++) {
			mask = harris(cv::Rect(i, j,mask_size,mask_size));
			minMaxLoc(mask,&min,&max,&min_loc,&max_loc);
			max_loc.x += i;
			max_loc.y += j;
			if(max >= val){
				points.push_back(KeyPoint(max_loc,1.f));
			}
        }
    }
	return points;
}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

int main(void){
    String path_in = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/part1/book/book1/MIX/day5/left/";
    String path_out = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/images_pgm/left/";
   
    vector<cv::String> img_name;
    glob(path_in + "*.jpg", img_name, false);  

    int n_imgs = img_name.size(); //number of png files in images folder

    Mat imgPrev;
    vector<Point2f> pointsPrev;
    vector<vector<Point2f> > allPoints;
    vector<vector<Point2f> > allPointsCalculated;
    for(int i = 0; i < n_imgs; ++i) {
        cout << i << endl;
        Mat imgActual = imread(img_name[i],CV_LOAD_IMAGE_GRAYSCALE);
        
        //Mat harris = harrisFilter(imgActual);
        //	Find Keypoints
        //vector<KeyPoint> keyPointsActual = getHarrisPoints(harris, 130); //120 120,120 130, 80 70,100 130
        //vector<Point2f> pointsActual; 
        //KeyPoint::convert(keyPointsActual, pointsActual);

        vector<KeyPoint> kp;
        vector<Point2f> pointsActual; 

        Ptr<ORB> detector = ORB::create();
        detector->detect(imgActual, kp);
        //std::cout << "Found " << kp.size() << " Keypoints " << std::endl;

        for(vector<KeyPoint>::iterator it=kp.begin(); it!=kp.end(); ++it){
            pointsActual.push_back(it->pt);
        }

        //	Draw KeyPoints
        //Mat img_keypoints;
        //drawKeypoints(imgActual, kp, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
        //imshow("img", img_keypoints);
        //waitKey(0);
        vector<uchar> featuresFound;
        vector<float> err;
        vector<Point2f> pointsCalculated; 
        //calcOpticalFlowFarneback(res_frame_gris_prev,res_frame_gris, None, 0.5, 3, 15, 3, 5, 1.2, 0);
        if(i != 0){
            //imshow("actual",imgActual);
            //waitKey(0);
            //imshow("prev",imgPrev);
            //waitKey(0);
            //cout << pointsActual.size()<< endl;
            //cout << pointsPrev.size()<< endl;
            calcOpticalFlowPyrLK(imgPrev, imgActual, pointsPrev,pointsCalculated,featuresFound,err);
            allPoints.push_back(pointsActual);
            allPointsCalculated.push_back(pointsCalculated);
            /*
            vector<KeyPoint> kpCalculated;
            KeyPoint::convert(pointsCalculated, kpCalculated, 10, 1, 0, -1);
            Mat img_keypoints_calculated;
            drawKeypoints(imgActual, kpCalculated, img_keypoints_calculated, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
            imshow("img_keypoints_calculated", img_keypoints_calculated);
            waitKey(0);

            Mat img_keypoints;
            drawKeypoints(imgActual, kp, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
            imshow("img_keypoints", img_keypoints);
            waitKey(0);
            */
        }
        imgPrev = imgActual;
        pointsPrev = pointsActual;
    }
    vector<Mat> allCov;
    vector<Mat> allMu;
    cout << "allPointsCalculated.size()" << endl;
    cout << allPointsCalculated.size() << endl;
    for(int i=0; i<allPointsCalculated.size(); i++){
        Mat cov, mu;   
        //Mat matAllPointsCalculated = (Mat_<float>(allPointsCalculated[i].size(),2) << allPointsCalculated[i]);
        //Mat matAllPointsCalculated = Mat(allPointsCalculated[i].size(),2,CV_8UC1);
        //memcpy(matAllPointsCalculated.data, allPointsCalculated[i].data(), allPointsCalculated[i].size()*sizeof(Point2f));
        //Mat matAllPointsCalculated(allPointsCalculated[i],CV_8UC1);
        vector<float> xCoord;
        vector<float> yCoord;
        for(int j=0; j<allPointsCalculated[i].size(); j++){
            float x = allPointsCalculated[i][j].x;
            float y = allPointsCalculated[i][j].y;
            xCoord.push_back(x);
            yCoord.push_back(y);
        }
        Mat matActualPointsCalculated; //These are the destination matrices 
        Mat matX(xCoord);
        Mat matY(yCoord);
        hconcat(matX, matY, matActualPointsCalculated);
        //cv::cvtColor(matAllPointsCalculated, matAllPointsCalculated, CV_RGB2GRAY);
        //cout << matAllPointsCalculated.row(0) << endl;
        //cout << matAllPointsCalculated.row(0).size() << endl;
        //cout << matAllPointsCalculated.channels() << endl;
        
        calcCovarMatrix(matActualPointsCalculated, cov, mu, CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE);    
        //cout << "cov" << endl;
        //cout << cov << endl;
        //cout << "mu" << endl;
        //cout << mu << endl;
        
        allCov.push_back(cov);
        allMu.push_back(mu);
    }

    vector<vector<float> > mahalanobisDistance;
    vector<vector<int> > viPosinFrames;
    for(int i = 0; i < allPointsCalculated.size(); i++) {
        Mat icov;
        invert(allCov[i], icov, DECOMP_SVD);
        icov.convertTo(icov,CV_32FC1);
        
        vector<float> mahalanobisDistanceActualPoints;
        vector<int> vi_pos;

        for(int j=0; j<allPointsCalculated[i].size(); j++){
            allMu[i].convertTo(allMu[i],CV_32FC1);

            Mat mu_i = allMu[i].t();
            Mat v;
            v.push_back(allPointsCalculated[i][j].x);
            v.push_back(allPointsCalculated[i][j].y);
    
            cout << "Mahalanobis(v, mu_i, icov)" << endl;
            cout << Mahalanobis(v, mu_i, icov) << endl;
            if(Mahalanobis(v, mu_i, icov) < 0.9){
                mahalanobisDistanceActualPoints.push_back(Mahalanobis(v, mu_i, icov));
                vi_pos.push_back(j);
            }
        }
        vector<KeyPoint> kpMahalanobis;
        vector<Point2f> pointsMahalanobis;
        for(int k=0; k<vi_pos.size(); k++){
            pointsMahalanobis.push_back(allPointsCalculated[i][vi_pos[k]]);
        }
        
        Mat imgActual = imread(img_name[i+1],CV_LOAD_IMAGE_GRAYSCALE); 
        
        KeyPoint::convert(pointsMahalanobis, kpMahalanobis, 10, 1, 0, -1);
        Mat img_keypoints_mahalanobis;
        drawKeypoints(imgActual, kpMahalanobis, img_keypoints_mahalanobis, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
        imshow("img_keypoints_mahalanobis", img_keypoints_mahalanobis);
        waitKey(0);

        vector<KeyPoint> kpReal;
        KeyPoint::convert(allPointsCalculated[i], kpReal, 10, 1, 0, -1);
        Mat img_keypoints_Real;
        drawKeypoints(imgActual, kpReal, img_keypoints_Real, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
        imshow("img_keypoints_Real", img_keypoints_Real);
        waitKey(0);


        //mahalanobisDistance.push_back(mahalanobisDistanceActualPoints);
        //viPosinFrames.push_back(vi_pos);
        
    }
    //calcCovarMatrix(samples, cov, mu, CV_COVAR_NORMAL | CV_COVAR_ROWS);

    return 0; 
}

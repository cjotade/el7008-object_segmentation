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
	for (int i=0 ; i<harris.cols-(mask_size-1) ; i++) {
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

void calcStatics(vector<Point2f> pointsCalculated, Mat& cov, Mat& mu){
    vector<float> xCoord;
    vector<float> yCoord;
    for(int j=0; j<pointsCalculated.size(); j++){
        float x = pointsCalculated[j].x;
        float y = pointsCalculated[j].y;
        xCoord.push_back(x);
        yCoord.push_back(y);
    }
    Mat matActualPointsCalculated; //These are the destination matrices 
    Mat matX(xCoord);
    Mat matY(yCoord);
    hconcat(matX, matY, matActualPointsCalculated);
    calcCovarMatrix(matActualPointsCalculated, cov, mu, CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE);    
}

void selectPointsMahalanobis(vector<Point2f> pointsCalculated,Mat& icov, Mat& mu, vector<int>& vi_pos, double threshold){
    for(int j=0; j<pointsCalculated.size(); j++){
        mu.convertTo(mu,CV_32FC1);
        Mat mu_i = mu.t();
        Mat v;
        v.push_back(pointsCalculated[j].x);
        v.push_back(pointsCalculated[j].y);

        cout << "Mahalanobis(v, mu_i, icov)" << endl;
        cout << Mahalanobis(v, mu_i, icov) << endl;
        if(Mahalanobis(v, mu_i, icov) < threshold){
            //mahalanobisDistanceActualPoints.push_back(Mahalanobis(v, mu_i, icov));
            vi_pos.push_back(j);
        }
    }
}

Mat findContours(Mat img){
    threshold(img,img, 100, 255, THRESH_BINARY_INV);
    RNG rng(12345);
    vector<vector<Point> > contours;
    findContours(img, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    vector<vector<Point> > contours_poly(contours.size());
    vector<Rect> boundRect(contours.size());
    Scalar color;
    Mat drawing = Mat::zeros(img.size(), CV_8UC3);
    
    for(size_t i = 0; i < contours.size(); i++){
        approxPolyDP(contours[i], contours_poly[i], 3, true); 
        boundRect[i] = boundingRect(contours_poly[i]);
    }
    for( size_t i = 0; i< contours.size(); i++ ){
        color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );

        Rect selectedRect;
        int boundArea = boundRect[i].area();
        int realArea = img.rows * img.cols;

        if(boundArea != realArea && boundArea >= 1000){
            cout << "boundArea realArea ENTER" << endl; //cout
            cout << boundArea << " " << realArea << endl;

            if(boundArea >=3000){
                selectedRect = boundRect[i];
                color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
                rectangle(drawing, selectedRect.tl(), selectedRect.br(),color, 2);
            }
        }
    }
    return drawing;
}

int main(void){
    String path_in = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/part1/book/book1/MIX/day5/left/";
    String path_out = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/proyectoRGB/contours/";
   
    vector<cv::String> img_name;
    glob(path_in + "*.jpg", img_name, false);  

    int n_imgs = img_name.size(); //number of png files in images folder

    Mat imgPrev;
    vector<Point2f> pointsPrev;
    for(int i = 0; i < n_imgs; ++i) {
        cout << i << endl;
        Mat imgActual = imread(img_name[i],CV_LOAD_IMAGE_GRAYSCALE);
        
        vector<KeyPoint> kp;
        vector<Point2f> pointsActual; 

        Ptr<ORB> detector = ORB::create();
        detector->detect(imgActual, kp);

        for(vector<KeyPoint>::iterator it=kp.begin(); it!=kp.end(); ++it){
            pointsActual.push_back(it->pt);
        }
    
        vector<uchar> featuresFound;
        vector<float> err;
        vector<Point2f> pointsCalculated; 
        if(i > 0){
            calcOpticalFlowPyrLK(imgPrev, imgActual, pointsPrev,pointsCalculated,featuresFound,err);
            Mat cov, mu;
            calcStatics(pointsCalculated,cov,mu);

            vector<vector<float> > mahalanobisDistance;
            vector<vector<int> > viPosinFrames;
            Mat icov;
            invert(cov, icov, DECOMP_SVD);
            icov.convertTo(icov,CV_32FC1);
            
            vector<int> vi_pos;
            
            selectPointsMahalanobis(pointsCalculated,icov,mu,vi_pos,0.7);

            vector<KeyPoint> kpMahalanobis;
            vector<Point2f> pointsMahalanobis;
            for(int k=0; k<vi_pos.size(); k++){
                pointsMahalanobis.push_back(pointsCalculated[vi_pos[k]]);
            }
            
            KeyPoint::convert(pointsMahalanobis, kpMahalanobis, 10, 1, 0, -1);
            Mat img_keypoints_mahalanobis;
            drawKeypoints(imgActual, kpMahalanobis, img_keypoints_mahalanobis, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
            imshow("img_keypoints_mahalanobis", img_keypoints_mahalanobis);
            waitKey(0);

            vector<KeyPoint> kpReal;
            KeyPoint::convert(pointsCalculated, kpReal, 10, 1, 0, -1);
            Mat img_keypoints_Real;
            drawKeypoints(imgActual, kpReal, img_keypoints_Real, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
            //imshow("img_keypoints_Real", img_keypoints_Real);
            //waitKey(0);

            Mat concat;
            hconcat(img_keypoints_Real,img_keypoints_mahalanobis,concat);
            //imshow("concat",concat);

            String img_out_name = img_name[i].substr(path_in.length(),img_name[i].length()-path_in.length()-4) + ".jpg";
            String path_save_out = path_out + img_out_name;
            //imwrite(path_save_out,concat);
            
            
            Mat drawing = findContours(imgActual);

            Mat result;
            addWeighted(img_keypoints_mahalanobis, 0.5, drawing, 0.5, 0.0, result);
            imshow("over",result);
            waitKey(0);
            imwrite(path_save_out,result);
            

            /*
            // declare Mat variables, thr, gray and src
            Mat thr, gray, src;
            
            // convert image to grayscale
            cvtColor( src, gray, COLOR_BGR2GRAY );
            
            // convert grayscale to binary image
            threshold( gray, thr, 100,255,THRESH_BINARY );
            
            // find moments of the image
            Moments m = moments(thr,true);
            Point p(m.m10/m.m00, m.m01/m.m00);
            
            // coordinates of centroid
            cout<< Mat(p)<< endl;
            
            // show the image with a point mark at the centroid
            circle(src, p, 5, Scalar(128,0,0), -1);
            imshow("Image with center",src);
            waitKey(0);
            */
        
        }   
        imgPrev = imgActual;
        pointsPrev = pointsActual;
    }

    return 0; 
}

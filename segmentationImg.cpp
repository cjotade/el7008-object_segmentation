#define _DEBUG

// Instruciones:
// Dependiendo de la versión de opencv, pueden cambiar los archivos .hpp a usar

#include <opencv2/opencv.hpp>

#include "opencv2/core/version.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>


#include <iostream>
#include <numeric>
#include <algorithm>

using namespace std;
using namespace cv;

void filterDisparity(Mat img){    
    double thresh = 50;
    double maxValue = 255;
    threshold(img,img, thresh, maxValue, THRESH_TOZERO);
    //imshow("thresh",img);
    //cout << img << endl;
    //waitKey(0);
    GaussianBlur(img, img, Size(5,5), 1.5, 1.5);
    for(int i=0; i<4; i++){
        dilate(img, img, Mat());
        GaussianBlur(img, img, Size(5,5), 2, 2);
    }
    for(int i=0; i<2; i++){
        erode(img,img, Mat());
        GaussianBlur(img, img, Size(5,5), 2, 2);
    }
}

void blobSelection(Mat img, Mat img_real){
    
    cout << img.size << endl;
    cv::cvtColor(img, img, CV_RGB2GRAY);
    //normalize(label, seeMyLabels, 0, 255, NORM_MINMAX, CV_8U);
    double min, max;
    Point min_loc, max_loc;

    minMaxLoc(img, &min, &max, &min_loc, &max_loc);
    //Mat connectLabels;
    threshold(img,img, 200, 255, THRESH_BINARY_INV);
    //bitwise_not(img,img);
    //connectedComponents(img,connectLabels,8,CV_32S);
    imshow("img",img);
    waitKey(0);

    
    // Set up the detector with default parameters.
    
    RNG rng(12345);
    vector<vector<Point> > contours;
    findContours(img, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    
    vector<vector<Point> > contours_poly(contours.size());
    vector<Rect> boundRect(contours.size());

    for(size_t i = 0; i < contours.size(); i++){
        approxPolyDP(contours[i], contours_poly[i], 3, true);
        boundRect[i] = boundingRect(contours_poly[i]);
        
    }

    Mat drawing = Mat::zeros(img.size(), CV_8UC3);

    for( size_t i = 0; i< contours.size(); i++ ){
        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        
        int width = abs(boundRect[i].tl().x - boundRect[i].br().x);
        int height = abs(boundRect[i].tl().y - boundRect[i].br().y);
        int boundArea = width*height;
        int realArea = img_real.rows * img_real.cols;

        if(boundArea != realArea && boundArea >= 5000){
            cout << boundArea << " " <<realArea << endl;
            rectangle(drawing, boundRect[i].tl(), boundRect[i].br(),color, 2);
            Rect selectedRect(boundRect[i]);
            Point2d centerRect(boundRect[i].tl().x+abs(boundRect[i].tl().x - boundRect[i].br().x)/2,boundRect[i].tl().y+abs(boundRect[i].tl().y - boundRect[i].br().y)/2);
            cout << centerRect << endl;
        }
    }
    Mat result;
    addWeighted(img_real, 0.5, drawing, 0.5, 0.0, result);
    //imshow("1", drawing );
    //imshow("2",img_in);
    imshow("over",result);
    waitKey(0);

    
}

void readDisparityImages(String path_in,String path_out,String path_real){
    vector<cv::String> img_name;
    glob(path_in + "*.pgm", img_name, false);  

    vector<cv::String> img_real_name;
    glob(path_real + "*.pgm", img_real_name, false);  

    int n_imgs = img_name.size(); //number of png files in images folder

    for(int i = 0; i < n_imgs; ++i) {
        cout << img_name[i] << endl;
        Mat img = imread(img_name[i]);
        Mat img_real = imread(img_real_name[i]);
        imshow("disparity_img",img);
        waitKey(0);
        filterDisparity(img);
        blobSelection(img,img_real);

        String img_path_out = img_name[i].substr(0,img_name[i].length()-4) + ".pgm";
        cout << img_path_out << endl;
        //imwrite(img_path_out,img);
    }
}


int main(void){
    String path_in = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/images_disp/";
    String path_out = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/images_disp/";
    String path_real = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/images_real/";
    readDisparityImages(path_in,path_out,path_real);

    return 0; 
}


/*

    //Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create();
    
    SimpleBlobDetector::Params params;
    params.filterByArea = true;
    params.minArea = 1;
    params.maxArea = 1000;
    params.filterByColor = true;
    params.blobColor = 255;

    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    std::vector<KeyPoint> keypoints;
    detector->detect( img, keypoints);
    
    // Draw detected blobs as red circles.
    // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
    Mat im_with_keypoints;
    drawKeypoints(img, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    Mat result_blob;
    addWeighted(img_real, 0.5, im_with_keypoints, 0.5, 0.0, result_blob);
    // Show blobs
    imshow("result blob", result_blob );
    waitKey(0);

    for(vector<KeyPoint>::iterator blobIterator = keypoints.begin(); blobIterator != keypoints.end(); blobIterator++){
        cout << "size of blob is: " << blobIterator->size << endl;
        cout << "point is at: " << blobIterator->pt.x << " " << blobIterator->pt.y << endl;
    
    }
    */ 
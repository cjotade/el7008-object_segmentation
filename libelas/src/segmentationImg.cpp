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

void buffer4imgs(Rect actualRect,vector<Rect> buffer_rect, int i_img, Point2d& meanPtTl, Point2d& meanPtBr){
    vector<double> w;
    double w0 = 0.1;
    double w1 = 0.3;
    double w2 = 0.5;
    double wActual = 0.1;
    w.push_back(w0);
    w.push_back(w1);
    w.push_back(w2);
    w.push_back(wActual);

    double sumTlx = 0; 
    double sumTly = 0;
    double sumBrx = 0; 
    double sumBry = 0;  
    int i = 0;
    for (int j = i_img-3; j < i_img; j++) {
        sumTlx += w[i]*(double)buffer_rect[j].tl().x;
        sumTly += w[i]*(double)buffer_rect[j].tl().y;
        sumBrx += w[i]*(double)buffer_rect[j].br().x;
        sumBry += w[i]*(double)buffer_rect[j].br().y;
        i+=1;
    }
    int meanTlx = (int)(w[i]*actualRect.tl().x + sumTlx);
    int meanTly = (int)(w[i]*actualRect.tl().y + sumTly);
    int meanBrx = (int)(w[i]*actualRect.br().x + sumBrx);
    int meanBry = (int)(w[i]*actualRect.br().y + sumBry);
    
    meanPtTl.x = meanTlx;
    meanPtTl.y = meanTly;
    meanPtBr.x = meanBrx;
    meanPtBr.y = meanBry;
}

void buffer3imgs(vector<Rect> buffer_rect, int i_img, Point2d& meanPtTl, Point2d& meanPtBr){
    vector<double> w;
    double w0 = 0.2;
    double w1 = 0.2;
    double w2 = 0.6;
    w.push_back(w0);
    w.push_back(w1);
    w.push_back(w2);

    double sumTlx = 0; 
    double sumTly = 0;
    double sumBrx = 0; 
    double sumBry = 0;  
    int i = 0;
    for (int j = i_img-3; j < i_img; j++) {
        //cout << w[i] << endl;
        sumTlx += w[i]*(double)buffer_rect[j].tl().x;
        sumTly += w[i]*(double)buffer_rect[j].tl().y;
        sumBrx += w[i]*(double)buffer_rect[j].br().x;
        sumBry += w[i]*(double)buffer_rect[j].br().y;
        i+=1;
    }
    //cout << "sums" << endl;
    //cout << sumTlx << endl;
    //cout << sumTly << endl;
    //cout << sumBrx << endl;
    //cout << sumBry << endl;
    int meanTlx = (int)sumTlx;
    int meanTly = (int)sumTly;
    int meanBrx = (int)sumBrx;
    int meanBry = (int)sumBry;
    
    meanPtTl.x = meanTlx;
    meanPtTl.y = meanTly;
    meanPtBr.x = meanBrx;
    meanPtBr.y = meanBry;
}

double IOU2Rects(Rect iboundingRectActual, Rect boundingRectPrev){
    Rect interRect = iboundingRectActual & boundingRectPrev;
    //cout << "interRect" << endl;
    //cout << interRect.area() << endl;
    Rect unionRect = iboundingRectActual | boundingRectPrev;
    //cout << "unionRect" << endl;
    //cout << unionRect.area() << endl;

	double iou = double(interRect.area()) / unionRect.area();
    cout << "iou" << endl;
    cout << iou << endl;
	return iou;
}

double IOU4Rects(Rect iboundingRectActual, vector<Rect>& buffer_rect, int i_img){
    Rect interRect = iboundingRectActual & buffer_rect[i_img-3] & buffer_rect[i_img-2] & buffer_rect[i_img-1];
    //cout << "interRect" << endl;
    //cout << interRect.area() << endl;
    Rect unionRect = iboundingRectActual | buffer_rect[i_img-3] | buffer_rect[i_img-2] | buffer_rect[i_img-1];
    //cout << "unionRect" << endl;
    //cout << unionRect.area() << endl;

	double iou = double(interRect.area()) / unionRect.area();
    cout << "iou" << endl;
    cout << iou << endl;
	return iou;
}


void filterDisparity(Mat& img){    
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

Mat blobSelection(Mat& img, Mat img_real,vector<Rect>& buffer_rect, int i_img){
    cout << "i_img" << endl;
    cout << i_img+1 << endl;
    cout << "img.size" << endl;
    cout << img.size << endl;
    cv::cvtColor(img, img, CV_RGB2GRAY);
   
    threshold(img,img, 200, 255, THRESH_BINARY_INV);
    //bitwise_not(img,img);
    //connectedComponents(img,connectLabels,8,CV_32S);
    //imshow("img",img);
    //waitKey(0);

    // Set up the detector with default parameters.
    
    RNG rng(12345);
    vector<vector<Point> > contours;
    findContours(img, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    
    vector<vector<Point> > contours_poly(contours.size());
    vector<Rect> boundRect(contours.size());
    cout << "boundRect.size()" << endl;
    cout << boundRect.size() << endl;
    for(size_t i = 0; i < contours.size(); i++){
        approxPolyDP(contours[i], contours_poly[i], 3, true); //ver esto 3
        boundRect[i] = boundingRect(contours_poly[i]);
    }

    Mat drawing = Mat::zeros(img.size(), CV_8UC3);

    //vector<Rect> first_buff_rect;
    Scalar color;
    vector<double> IOUs;
    vector<int> posBound;
    cout << "contours.size()" << endl;
    cout << contours.size() << endl;
    for( size_t i = 0; i< contours.size(); i++ ){
        color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        int boundArea = boundRect[i].area();
        int realArea = img_real.rows * img_real.cols;
        cout << "boundArea realArea" << endl;
        cout << boundArea << " " << realArea << endl;
        if(boundArea != realArea && boundArea >= 1000){
            cout << "boundArea realArea ENTER" << endl;
            cout << boundArea << " " << realArea << endl;
            if(i_img < 3 && boundArea >=3000){
                Rect selectedRect(boundRect[i]);
                buffer_rect.push_back(selectedRect);
                cout << "bufer_rect.size()" << endl;
                cout << buffer_rect.size() << endl;
                rectangle(drawing, selectedRect.tl(), selectedRect.br(),color, 2);
            }
            else if (i_img >= 3){
                //calc IOU para cada boundingRect[i] y guardo en vector
                IOUs.push_back(IOU2Rects(boundRect[i],buffer_rect[i_img-1]));
                posBound.push_back(i);
            }
        }
        
    }
    if(i_img >= 3){
        Point2d meanPtTl;
        Point2d meanPtBr;
        int posMaxIOU = std::distance(IOUs.begin(), std::max_element(IOUs.begin(), IOUs.end()));
        double maxIOU = *max_element(IOUs.begin(), IOUs.end());
        //cout << "posMaxIOU" << endl;
        //cout << posMaxIOU << endl;
        cout << "maxIOU" << endl;
        cout << maxIOU << endl;
        Rect selectedRect;
        Rect actualRect = boundRect[posBound[posMaxIOU]];
        bool minArea = abs(buffer_rect[i_img-1].area() - actualRect.area()) < 20000;
        cout << "minArea" << endl;
        cout << abs(buffer_rect[i_img-1].area() - actualRect.area()) << endl;

        //elegir como condicion comparacion ious entre promedio 3 anteriores
        double compareIOUs = IOU4Rects(actualRect,buffer_rect,i_img);
        if(compareIOUs >= 0.3 && minArea){ //and areas
            cout << "MAX IOU" << endl;
            //elige de todos los bound actuales el que tenga mayor IOU con respecto del anterior
            selectedRect = actualRect;
        }
        else{
            if(buffer_rect[i_img-1].area() > actualRect.area()){
                cout << "BUFFER4" << endl;
                buffer4imgs(actualRect,buffer_rect,i_img,meanPtTl,meanPtBr);
            }
            else{
                cout << "BUFFER3" << endl;
                buffer3imgs(buffer_rect,i_img,meanPtTl,meanPtBr);
            }
            selectedRect = Rect(meanPtTl,meanPtBr);
        }
        buffer_rect.push_back(selectedRect);
        rectangle(drawing, selectedRect.tl(), selectedRect.br(),color, 2);
        //asignar el boundingRect que tenga mayor IOU del vector y buffer entre 3 anteriores
        
    }
    Mat result;
    addWeighted(img_real, 0.5, drawing, 0.5, 0.0, result);
    imshow("over",result);
    waitKey(0);

    return result;
}



void readDisparityImages(String path_in,String path_out,String path_real){
    vector<cv::String> img_name;
    glob(path_in + "*.pgm", img_name, false);  

    vector<cv::String> img_real_name;
    glob(path_real + "*.jpg", img_real_name, false);  

    int n_imgs = img_name.size(); //number of png files in images folder

    vector<Rect> buffer_rect;
    for(int i = 0; i < n_imgs; ++i) {
        cout << img_name[i] << endl;
        Mat img = imread(img_name[i]);
        //cout << "watafas0" << endl;
        //cout << img_real_name[i] << endl;
        Mat img_real = imread(img_real_name[i]);
        //cout << "watafas" << endl;
        //imshow("disparity_img",img);
        //waitKey(0);
        filterDisparity(img);
        Mat blob;
        blob = blobSelection(img,img_real,buffer_rect,i);

        String img_out_name = img_name[i].substr(path_in.length(),img_name[i].length()-path_in.length()-4) + ".jpg";
        //cout << "ke waaa" << endl;
        //cout << img_out_name << endl;
        String path_save_out = path_out + img_out_name;
        //cout << "FIN" << endl;
        //cout << path_save_out << endl;
        
        //cv::cvtColor(blob, blob, CV_RGB2GRAY);
        //std::vector<int> compression_params;
        //compression_params.push_back(CV_IMWRITE_PXM_BINARY);
        //compression_params.push_back(1);
        //imwrite(path_save_out,blob,compression_params);
        imwrite(path_save_out,blob);
    }
}


int main(void){
    String path_in = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/images_disp/images_pgm/left/";
    String path_out = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/images_segmentation/left/";
    String path_real = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/images_real/left/";
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
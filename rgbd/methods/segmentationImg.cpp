#define _DEBUG

// Instruciones:
// Dependiendo de la versión de opencv, pueden cambiar los archivos .hpp a usar

#include <opencv2/opencv.hpp>

#include "opencv2/core/version.hpp"
#include <opencv2/opencv.hpp>
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
        sumTlx += w[i]*(double)buffer_rect[j].tl().x;
        sumTly += w[i]*(double)buffer_rect[j].tl().y;
        sumBrx += w[i]*(double)buffer_rect[j].br().x;
        sumBry += w[i]*(double)buffer_rect[j].br().y;
        i+=1;
    }
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
    Rect unionRect = iboundingRectActual | boundingRectPrev;
	double iou = double(interRect.area()) / unionRect.area();
	return iou;
}

double IOU4Rects(Rect iboundingRectActual, vector<Rect>& buffer_rect, int i_img){
    Rect interRect = iboundingRectActual & buffer_rect[i_img-3] & buffer_rect[i_img-2] & buffer_rect[i_img-1];
    Rect unionRect = iboundingRectActual | buffer_rect[i_img-3] | buffer_rect[i_img-2] | buffer_rect[i_img-1];
	double iou = double(interRect.area()) / unionRect.area();
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

inline Point calcPoint(Point2f center, double R, double angle){
    return center + Point2f((float)cos(angle), (float)-sin(angle))*(float)R;
}

void kalmanTracking(Point2d centerActual, Point2d centerPrev, Point& measPt, Point& statePt, Point& predictPt, vector<Point>& mousev,vector<Point>& kalmanv, KalmanFilter KF){
    //KalmanFilter KF(4, 2, 0);
    Mat_<float> state(4, 1); /* (x, y, Vx, Vy) */
    Mat processNoise(4, 1, CV_32F);
    Mat_<float> measurement(2,1); measurement.setTo(Scalar(0));

    KF.statePre.at<float>(0) = centerPrev.x;
    KF.statePre.at<float>(1) = centerPrev.y;
    KF.statePre.at<float>(2) = 0;
    KF.statePre.at<float>(3) = 0;
    //KF.transitionMatrix = (Mat_<float>(4, 4) << 1,0,0,0,   0,1,0,0,  0,0,1,0,  0,0,0,1);
    KF.transitionMatrix = (Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(KF.errorCovPost, Scalar::all(.1));

    
    // First predict, to update the internal statePre variable
    Mat prediction = KF.predict();
    predictPt.x = prediction.at<float>(0);
    predictPt.y = prediction.at<float>(1);
    cout << "prediccion" << endl;
    cout << prediction << endl;
                
    // Get mouse point
    //measurement(0) = mouse_info.x;
    //measurement(1) = mouse_info.y;
    measurement(0) = centerActual.x;
    measurement(1) = centerActual.y;
                
    measPt.x = measurement(0);
    measPt.y = measurement(1);
    
    // The "correct" phase that is going to use the predicted value and our measurement
    Mat estimated = KF.correct(measurement);
    statePt.x = estimated.at<float>(0);
    statePt.y = estimated.at<float>(1);

    mousev.push_back(measPt);
    kalmanv.push_back(statePt);
}

/**
 * Finds the countors of given img
 * 
 */
vector<Rect> findContours(Mat img, vector<vector<Point> >& contours){
    // Set up the detector with default parameters and find contours.
    findContours(img, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    vector<vector<Point> > contours_poly(contours.size());
    vector<Rect> boundRect(contours.size());

    cout << "boundRect.size()" << endl; //cout
    cout << boundRect.size() << endl;

    for(size_t i = 0; i < contours.size(); i++){
        approxPolyDP(contours[i], contours_poly[i], 3, true); 
        boundRect[i] = boundingRect(contours_poly[i]);
    }
    return boundRect;
}

/**
 * Selects if the actualRect is a good option, else
 * Selects between buffer3 or buffer4
 * 
 */ 
Rect selectMethod(Rect actualRect,vector<Rect>& buffer_rect, int i_img,Point2d meanPtTl, Point2d meanPtBr, double compareIOU4Rects,double compareIOU2Rects, double minArea){
    if((compareIOU4Rects >= 0.3 || compareIOU2Rects >= 0.43) && minArea < 20000){ //and areas
        cout << "" << endl;
        cout << "actualRect ES SUFICIENTEMENTE BUENO" << endl;
        return actualRect;
    }
    else{
        cout << "actualRect.area()" << endl;
        cout << actualRect.area() << endl;

        if(buffer_rect[i_img-1].area() > actualRect.area() && actualRect.area() >= 20000){
            cout << "" << endl;
            cout << "BUFFER4" << endl;
            buffer4imgs(actualRect,buffer_rect,i_img,meanPtTl,meanPtBr);
        }
        else{
            cout << "" << endl;
            cout << "BUFFER3" << endl;
            buffer3imgs(buffer_rect,i_img,meanPtTl,meanPtBr);
        }
        return Rect(meanPtTl,meanPtBr);
    }
}

Rect selectActualRect(Rect selectedRect, vector<Rect> boundRect,vector<Rect>& buffer_rect, vector<double> IOUs,vector<int> posBound, int i_img){
    cout << "ENTER SELECTED AREA" << endl;
    Point2d meanPtTl;
    Point2d meanPtBr;
    Rect actualRect;
    if(IOUs.size() != 0){ // and maxIOU
        int posMaxIOU = std::distance(IOUs.begin(), std::max_element(IOUs.begin(), IOUs.end()));
        double maxIOU = *max_element(IOUs.begin(), IOUs.end());

        cout << "Seleccion por MAXIOU" << endl; //cout
        cout << maxIOU << endl;
        actualRect = boundRect[posBound[posMaxIOU]];
    }
    else{
        cout << "Seleccion por imagen anterior" << endl;
        actualRect = buffer_rect[i_img-1]; // VERIFICAR
    }
    return actualRect;
}

/**
 * Blob selection for a given disparity img
 * 
 * Note: its necesary that the result of the 3 frames contains just 1 roi (and a good one)
 *       if not, try to no use these img as first (just delete it)
 * 
 * @param Mat img                   the process image to find the blob and roi
 * @param Mat img_real              the real img for overlap with the roi
 * @param vector<Rect> buffer_rect  the vector of selectedRect from every frame
 * @param int i_img                 the ith frame
 * 
 * @return the resulting img weighted by the roi finded and the real img
 *  
*/
Mat blobSelection(Mat& img, Mat img_real,vector<Rect>& buffer_rect, int i_img, KalmanFilter KF,vector<Point>& mousev,vector<Point>& kalmanv){
    /**
     * Some variables you need to understand
     * 
     * @var vector<Rect> boundRect     contain all found rois for frame i_img
     * @var vector<double> IOUs        contain the IOUs between the boundRect[i] and the selected Rect in the previous frame (i_img-1)
     * @var vector<int> posBound       contain the position of i for post-calculus of the max IOU in boundRect
     * @var Rect selectedRect          the selected Rect for frame i_img
    */
    cout << "image number" << endl;
    cout << i_img+1 << endl;
    cout << "img.size" << endl;
    cout << img.size << endl;

    cv::cvtColor(img, img, CV_RGB2GRAY);
    threshold(img,img, 200, 255, THRESH_BINARY_INV);
    
    RNG rng(12345);
    vector<vector<Point> > contours;
    Mat drawing = Mat::zeros(img.size(), CV_8UC3);
    Scalar color;
    vector<double> IOUs;
    vector<int> posBound;
    //vector<Point> mousev,kalmanv;
    Rect selectedRect;

    vector<Rect> boundRect = findContours(img,contours);

    cout << "contours.size()" << endl; //cout
    cout << contours.size() << endl;
    cout << "SELECCION MEJOR BOUND" << endl;

    for(size_t i = 0; i< contours.size(); i++){
        color = Scalar(rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256));
        int boundArea = boundRect[i].area();
        int realArea = img_real.rows * img_real.cols;

        //First Filter of bounds
        if(boundArea != realArea && boundArea >= 1000){
            cout << "boundArea realArea ENTER" << endl; //cout
            cout << boundArea << " " << realArea << endl;

            //Second Filter of bounds for i_img < 3
            if(i_img < 3 && boundArea >=3000){
                selectedRect = boundRect[i];
                buffer_rect.push_back(selectedRect);

                cout << "bufer_rect.size()" << endl; //cout
                cout << buffer_rect.size() << endl;
                rectangle(drawing, selectedRect.tl(), selectedRect.br(),color, 2);
            }
            else if (i_img >= 3){
                cout << "IOU2RECTS" << endl;
                cout << IOU2Rects(boundRect[i],buffer_rect[i_img-1]) << endl;

                //if boundRect[i] is sufficient good, then select it
                //if(IOU2Rects(boundRect[i],buffer_rect[i_img-1]) > 0.5){
                if(IOU2Rects(boundRect[i],buffer_rect[i_img-1]) > 0.5){
                    cout << "" << endl;
                    cout << "ENTER BOUND SELECTED RECT LISTO" << endl;

                    selectedRect = boundRect[i];

                    cout << selectedRect.area() << endl;
                }
                //dont't select boundRect[i]
                else{ 
                    // FALTA AÑADIR UNA CONDICION BUENA, HASTA AHORA SIRVE PERO SERIA IDEAL UNA
                    cout << "" << endl;
                    cout << "ENTEr JUST IN IOUs" << endl;
                    //calc IOU para cada boundingRect[i] y guardo en vector
                }
                //Storage IOUT of boundRect[i]
                IOUs.push_back(IOU2Rects(boundRect[i],buffer_rect[i_img-1]));
                posBound.push_back(i);
            }
        }
    }
    /**
     * From here the method select if a good selectRect is "selected",
     * if not, see in the IOUs vector and extract the max one as actualRect.
     * So then compare the actualRect with the previous frames using IOU2Rects and IOU4Rects:
     * if the actualRect is good enought, the selectedRect gonna be equals to actualRect,
     * otherwise the method gonna select if it is better to use the buffer3 or the buffer4 
     * in order to select the selectedRect
     * 
     * @var Rect actualRect     the Rect with max IOU from IOUs vector or buffer_rect[i_img-1]
    */
    cout << "" << endl;
    cout << "SELECCION ACTUAL O BUFFERS" << endl;
    cout << "IOUs.size()" << endl;
    cout << IOUs.size() << endl;

    if(i_img >= 3){
        cout << "selectedRect.area()" << endl;
        cout << selectedRect.area() << endl;
        
        /**
         * selectedRect.area() != 0 means that boundRect[i] pass the IOU above filter,
         * so is no need to re-select it by the buffers or when by the maximum IOU 
         */
        if(selectedRect.area() == 0){
            Point2d meanPtTl;
            Point2d meanPtBr;

            Rect actualRect = selectActualRect(selectedRect,boundRect,buffer_rect,IOUs,posBound,i_img);
            double minArea = abs(buffer_rect[i_img-1].area() - actualRect.area());

            cout << "minArea" << endl; //cout
            cout << abs(buffer_rect[i_img-1].area() - actualRect.area()) << endl;

            //elegir como condicion comparacion ious entre promedio 3 anteriores
            double compareIOU4Rects = IOU4Rects(actualRect,buffer_rect,i_img);
            double compareIOU2Rects = IOU2Rects(actualRect,buffer_rect[i_img-1]);

            selectedRect = selectMethod(actualRect,buffer_rect,i_img,meanPtTl,meanPtBr,compareIOU4Rects,compareIOU2Rects,minArea);
        }
        /**
         * Here begins the Kalman filter approach taking the centerActual and the centerPrev
         * 
         * @var Point2d centerActual    the actual center of the selectedRect
         * @var Point2d centerPrev      the previous center of the previous selectedRect (buffer_rect[i_img-1])
         * 
         * @see kalmanTracking
        */
        buffer_rect.push_back(selectedRect);
        Point2d centerActual(buffer_rect[i_img].tl().x+abs(buffer_rect[i_img].tl().x - buffer_rect[i_img].br().x)/2,buffer_rect[i_img].tl().y+abs(buffer_rect[i_img].tl().y - buffer_rect[i_img].br().y)/2);
        Point2d centerPrev(buffer_rect[i_img-1].tl().x+abs(buffer_rect[i_img-1].tl().x - buffer_rect[i_img-1].br().x)/2,buffer_rect[i_img-1].tl().y+abs(buffer_rect[i_img-1].tl().y - buffer_rect[i_img-1].br().y)/2);
        Point measPt;
        Point statePt;
        Point predictPt;
        kalmanTracking(centerActual, centerPrev, measPt, statePt, predictPt, mousev, kalmanv, KF);
        cout << "POINTSSSSSS" << endl;
        cout << measPt.x << " " << measPt.y << endl;
        cout << statePt.x << " " << statePt.y << endl;
        cout << predictPt.x << " " << predictPt.y << endl;
        rectangle(drawing, selectedRect.tl(), selectedRect.br(),color, 2);
        //circle(drawing,statePt,2,Scalar(255,255,255));
        //circle(drawing,measPt,2,Scalar(0,0,255));
        //circle(drawing,predictPt,2,Scalar(0,255,0));
        
        #define drawCross( center, color, d )                                           \
            line( drawing, Point( center.x - d, center.y - d ),                         \
                            Point( center.x + d, center.y + d ), color, 1, LINE_AA, 0); \
            line( drawing, Point( center.x + d, center.y - d ),                         \
                            Point( center.x - d, center.y + d ), color, 1, LINE_AA, 0 )
        
        drawCross( statePt, Scalar(255,255,255), 5 );
        drawCross( measPt, Scalar(0,0,255), 5 );
        drawCross( predictPt, Scalar(0,255,0), 5);
    
        for (int i = 0; i < mousev.size()-1; i++){
            line(drawing, mousev[i], mousev[i+1], Scalar(255,255,0), 1);
            //cout << "wat" << endl;
        }
        
        for (int i = 0; i < kalmanv.size()-1; i++){
            line(drawing, kalmanv[i], kalmanv[i+1], Scalar(0,155,255), 1);
            //cout << "wat1" << endl;
        }
        //drawCross( statePt, Scalar(255,255,255), 3 );
        //drawCross( measPt, Scalar(0,0,255), 3 );
        //drawCross( predictPt, Scalar(0,255,0), 3 );
        //line( img, statePt, measPt, Scalar(0,0,255), 3, LINE_AA, 0 );
        //line( img, statePt, predictPt, Scalar(0,255,255), 3, LINE_AA, 0 );
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

    int n_imgs = img_name.size(); //number of png files in images folder

    vector<Rect> buffer_rect;
    KalmanFilter KF(4, 2, 0);
    vector<Point> mousev,kalmanv;
    for(int i = 0; i < n_imgs; ++i) {
        String img_real_name = path_real + img_name[i].substr(path_in.length(),img_name[i].length() - path_in.length() -4) + ".jpg";

        cout << img_name[i] << endl;  
        cout << img_real_name << endl;  
        
        Mat img = imread(img_name[i]);
        Mat img_real = imread(img_real_name);
        filterDisparity(img);
        cout << "filter?" << endl;
        Mat blob;
        blob = blobSelection(img,img_real,buffer_rect,i,KF,mousev,kalmanv);
        std::cout << "blob?" << endl;

        String img_out_name = img_name[i].substr(path_in.length(),img_name[i].length()-path_in.length()-4) + ".jpg";
        String path_save_out = path_out + img_out_name;
        imwrite(path_save_out,blob);
    }
}


int main(void){
    /**
     * Change paths
     * 
     * @var String path_in      the path where are the disparity images
     * @var String path_out     the path where you want to save the images with the ROI
     * @var String path_real    the path where are the real images
     */

    //String path_in = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/images_disp/images_pgm/left/";
    //String path_out = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/images_segmentation/left/";
    //String path_real = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/images_real/left/";
    String path_in = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/IMAGES MIX/images_disp/images_pgm/left/";
    String path_out = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/OUTPUT/";
    String path_real = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/IMAGES MIX/images_real/left/";

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


   /*
void kalmanTracker(){
    Mat img(500, 500, CV_8UC3);
    KalmanFilter KF(2, 1, 0);
    Mat state(2, 1, CV_32F); // (phi, delta_phi) 
    Mat processNoise(2, 1, CV_32F);
    Mat measurement = Mat::zeros(1, 1, CV_32F);
    char code = (char)-1;

    for(;;)
    {
        randn( state, Scalar::all(0), Scalar::all(0.1) );
        KF.transitionMatrix = (Mat_<float>(2, 2) << 1, 1, 0, 1);

        setIdentity(KF.measurementMatrix);
        setIdentity(KF.processNoiseCov, Scalar::all(1e-5));
        setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
        setIdentity(KF.errorCovPost, Scalar::all(1));

        randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));

        for(;;)
        {
            Point2f center(img.cols*0.5f, img.rows*0.5f);
            float R = img.cols/3.f;
            double stateAngle = state.at<float>(0);
            Point statePt = calcPoint(center, R, stateAngle);

            Mat prediction = KF.predict();
            double predictAngle = prediction.at<float>(0);
            Point predictPt = calcPoint(center, R, predictAngle);

            randn( measurement, Scalar::all(0), Scalar::all(KF.measurementNoiseCov.at<float>(0)));

            // generate measurement
            measurement += KF.measurementMatrix*state;

            double measAngle = measurement.at<float>(0);
            Point measPt = calcPoint(center, R, measAngle);

            // plot points
            #define drawCross( center, color, d )                                        \
                line( img, Point( center.x - d, center.y - d ),                          \
                             Point( center.x + d, center.y + d ), color, 1, LINE_AA, 0); \
                line( img, Point( center.x + d, center.y - d ),                          \
                             Point( center.x - d, center.y + d ), color, 1, LINE_AA, 0 )

            img = Scalar::all(0);
            drawCross( statePt, Scalar(255,255,255), 3 );
            drawCross( measPt, Scalar(0,0,255), 3 );
            drawCross( predictPt, Scalar(0,255,0), 3 );
            line( img, statePt, measPt, Scalar(0,0,255), 3, LINE_AA, 0 );
            line( img, statePt, predictPt, Scalar(0,255,255), 3, LINE_AA, 0 );

            if(theRNG().uniform(0,4) != 0)
                KF.correct(measurement);

            randn( processNoise, Scalar(0), Scalar::all(sqrt(KF.processNoiseCov.at<float>(0, 0))));
            state = KF.transitionMatrix*state + processNoise;

            imshow( "Kalman", img );
            code = (char)waitKey(100);

            if( code > 0 )
                break;
        }
        if( code == 27 || code == 'q' || code == 'Q' )
            break;
    }
}
*/
/*
void kalmanTracking(Point2d centerActual, Point2d centerPrev, Point& measPt, Point2d& statePt){
    KalmanFilter KF(4, 2, 0);
    KF.transitionMatrix = (Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
    Mat_<float> measurement(2,1); measurement.setTo(Scalar(0));
    
    // init...
    //KF.statePre.at<float>(0) = mouse_info.x;
    KF.statePre.at<float>(0) = centerPrev.x;
    //KF.statePre.at<float>(1) = mouse_info.y;
    KF.statePre.at<float>(1) = centerPrev.y;
    KF.statePre.at<float>(2) = 0;
    KF.statePre.at<float>(3) = 0;
    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(KF.errorCovPost, Scalar::all(.1));

    // First predict, to update the internal statePre variable
    Mat prediction = KF.predict();
    Point predictPt(prediction.at<float>(0),prediction.at<float>(1));
                
    // Get mouse point
    //measurement(0) = mouse_info.x;
    measurement(0) = centerActual.x;
    //measurement(1) = mouse_info.y;
    measurement(1) = centerActual.y;
                
    measPt.x = measurement(0);
    measPt.y = measurement(1);
    
    // The "correct" phase that is going to use the predicted value and our measurement
    Mat estimated = KF.correct(measurement);
    statePt.x = estimated.at<float>(0);
    statePt.y = estimated.at<float>(1);
}
*/


/*
    // Set up the detector with default parameters and find contours.
    RNG rng(12345);
    vector<vector<Point> > contours;
    findContours(img, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    vector<vector<Point> > contours_poly(contours.size());
    vector<Rect> boundRect(contours.size());

    cout << "boundRect.size()" << endl; //cout
    cout << boundRect.size() << endl;

    for(size_t i = 0; i < contours.size(); i++){
        approxPolyDP(contours[i], contours_poly[i], 3, true); 
        boundRect[i] = boundingRect(contours_poly[i]);
    }
    */
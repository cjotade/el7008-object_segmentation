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

void calcStatics(vector<Point2f> pointsVel, Mat& cov, Mat& mu){
    vector<float> xCoord;
    vector<float> yCoord;
    for(int j=0; j<pointsVel.size(); j++){
        float x = pointsVel[j].x;
        float y = pointsVel[j].y;
        xCoord.push_back(x);
        yCoord.push_back(y);
    }
    Mat matActualPointsVel; //These are the destination matrices 
    Mat matX(xCoord);
    Mat matY(yCoord);
    hconcat(matX, matY, matActualPointsVel);
    calcCovarMatrix(matActualPointsVel, cov, mu, CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE);
    mu.convertTo(mu,CV_32FC1);
    cov.convertTo(cov,CV_32FC1);    
}

vector<int> selectPosPointsMahalanobis(vector<Point2f> pointsVel,Mat& icov, Mat& mu, double threshold){
    vector<int> vi_pos;
    for(int j=0; j<pointsVel.size(); j++){
        //mu.convertTo(mu,CV_32FC1);
        Mat mu_i = mu.t();
        Mat v;
        v.push_back(pointsVel[j].x);
        v.push_back(pointsVel[j].y);

        //cout << "Mahalanobis(v, mu_i, icov)" << endl;
        //cout << Mahalanobis(v, mu_i, icov) << endl;
        if(Mahalanobis(v, mu_i, icov) < threshold){
            vi_pos.push_back(j);
        }
    }
    return vi_pos;
}

vector<Point2f> egoMotion(Mat imgPrev, Mat imgActual, vector<Point2f> pointsPrev, vector<Point2f> pointsCalculated, vector<uchar> featuresFound,vector<float> err, double threshMahalanobis){
    vector<Point2f> pointsMahalanobis;

    calcOpticalFlowPyrLK(imgPrev, imgActual, pointsPrev,pointsCalculated,featuresFound,err);
    Mat cov, mu;
    
    calcStatics(pointsCalculated,cov,mu);
    
    Mat icov;
    invert(cov, icov, DECOMP_SVD);
    icov.convertTo(icov,CV_32FC1);

    //vector<int> vi_pos = selectPosPointsMahalanobis(pointsVel,icov,predMu,0.14);
    vector<int> vi_pos = selectPosPointsMahalanobis(pointsCalculated,icov,mu,threshMahalanobis);

    for(int k=0; k<vi_pos.size(); k++){
        pointsMahalanobis.push_back(pointsCalculated[vi_pos[k]]);
    }

    return pointsMahalanobis;
}

vector<Point2f> detectAndComputePoints(Mat imgActual, bool doORB){
    vector<KeyPoint> kp;
    vector<Point2f> pointsActual;

    if(doORB){
        Ptr<ORB> detector = ORB::create();
        detector->detect(imgActual, kp);

        for(vector<KeyPoint>::iterator it=kp.begin(); it!=kp.end(); ++it){
            pointsActual.push_back(it->pt);
        }
    }
    else{
        /// Parameters for Shi-Tomasi algorithm
        //vector<Point2f> pointsActual;
        double qualityLevel = 0.01;
        double minDistance = 10;
        int blockSize = 3;
        bool useHarrisDetector = false;
        double k = 0.04;
        int maxCorners = 200;
        /// Apply corner detection
        goodFeaturesToTrack( imgActual,
                    pointsActual,
                    maxCorners,
                    qualityLevel,
                    minDistance,
                    Mat(),
                    blockSize,
                    useHarrisDetector,
                    k );
    }

    return pointsActual;
}


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

void buffer2imgs(Rect actualRect,vector<Rect> buffer_rect, int i_img, Point2d& meanPtTl, Point2d& meanPtBr){
    vector<double> w;
    double w0 = 0.3;
    double wActual = 0.7;
    w.push_back(w0);
    w.push_back(wActual);

    double sumTlx = 0; 
    double sumTly = 0;
    double sumBrx = 0; 
    double sumBry = 0;  
    int i = 0;
    for (int j = i_img-1; j < i_img; j++) {
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
    //cout << "prediccion" << endl;
    //cout << prediction << endl;
                
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
Rect selectMethod(Rect actualRect,vector<Rect>& buffer_rect, int i_img,Point2d meanPtTl, Point2d meanPtBr, double compareIOU4Rects,double compareIOU2Rects, double minArea, int& enterBuffer, vector<Point2f> pointsMahalanobis){
    if((compareIOU4Rects >= 0.3 || compareIOU2Rects >= 0.43) && minArea < 20000){ //and areas
        enterBuffer = 0;
        cout << "enterBuffer COMPARE IOUS" << endl;
        cout << enterBuffer << endl;
        cout << "" << endl;
        cout << "======actualRect ES SUFICIENTEMENTE BUENO====" << endl;
        return actualRect;
    }
    else{
        cout << " " << endl;
        cout << "actualRect.area()" << endl;
        cout << actualRect.area() << endl;
        
        Rect selectedRect;
        
        cout << "buffer_rect[i_img-1].area()" << endl;
        cout << buffer_rect[i_img-1].area() << endl;
        //if(buffer_rect[i_img-1].area() > actualRect.area() && actualRect.area() >= 20000 && enterBuffer < 3){
        if(buffer_rect[i_img-1].area() > actualRect.area() && enterBuffer < 4){
            enterBuffer += 1;
            cout << "enterBuffer" << endl;
            cout << enterBuffer << endl;
            cout << "" << endl;
            cout << "BUFFER4" << endl;
            buffer4imgs(actualRect,buffer_rect,i_img,meanPtTl,meanPtBr);
            selectedRect = Rect(meanPtTl,meanPtBr);
        }
        else if (buffer_rect[i_img-1].area() <= actualRect.area() && enterBuffer < 4){
            enterBuffer += 1;
            cout << "enterBuffer" << endl;
            cout << enterBuffer << endl;
            cout << "" << endl;
            cout << "BUFFER3" << endl;
            buffer3imgs(buffer_rect,i_img,meanPtTl,meanPtBr);
            selectedRect = Rect(meanPtTl,meanPtBr);
        }
        else{
            cout << "enterBuffer" << endl;
            cout << enterBuffer << endl;
            enterBuffer = 0;
            cout << "" << endl;
            cout << "==============MOTION==============" << endl;
            //Rect pointsRect = boundingRect(pointsMahalanobis);
            //buffer2imgs(pointsRect,buffer_rect,i_img,meanPtTl,meanPtBr);
            //selectedRect = Rect(meanPtTl,meanPtBr);
            selectedRect = boundingRect(pointsMahalanobis);
        }
        return selectedRect;      
        
    }
}

Rect selectActualRect(Rect selectedRect, vector<Rect> boundRect,vector<Rect>& buffer_rect, vector<double> IOUs,vector<int> posBound, int i_img, vector<Point2f> pointsMahalanobis){
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
        cout << "=============================================================================" << endl;
        //actualRect = buffer_rect[i_img-1]; // VERIFICAR
        Rect pointsRect = boundingRect(pointsMahalanobis);
        Point2d mPtTl;
        Point2d mPtBr;
        buffer4imgs(pointsRect,buffer_rect,i_img,mPtTl,mPtBr);
        actualRect = Rect(mPtTl,mPtBr);
        //IOUs.push_back(IOU2Rects(actualRect,buffer_rect[i_img-1]));
    }
    return actualRect;
}

vector<Point2f> calcOpticalFlow2points(Mat imgPrev, Mat imgActual, vector<Point2f> pointsPrev){
    vector<uchar> featuresFound; 
    vector<float> err;
    vector<Point2f> pointsCalculated;
    calcOpticalFlowPyrLK(imgPrev, imgActual, pointsPrev,pointsCalculated,featuresFound,err);
    return pointsCalculated;
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
Mat blobSelection(Mat& img, Mat img_real,vector<Rect>& buffer_rect, int i_img, KalmanFilter KF,vector<Point>& mousev,vector<Point>& kalmanv, int& enterBuffer,vector<Point2f> pointsMahalanobis, Mat imgPrev,Mat imgActual){
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
    //Mat drawing = Mat::zeros(img.size(), CV_8UC3);
    Scalar color;
    vector<double> IOUs;
    vector<int> posBound;
    //vector<Point> mousev,kalmanv;
    Rect selectedRect;

    vector<Rect> boundRect = findContours(img,contours);

    cout << "contours.size()" << endl; //cout
    cout << contours.size() << endl;

    cout << " " << endl;
    cout << "SELECCION MEJOR BOUND" << endl;

    for(size_t i = 0; i< contours.size(); i++){
        cout << " "<< endl;
        cout << i << "COUNTOUR NUMBER" << endl;
        color = Scalar(rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256));
        int boundArea = boundRect[i].area();
        int realArea = img_real.rows * img_real.cols;

        //First Filter of bounds
        if(boundArea != realArea && boundArea >= 1000){
            cout << "boundArea realArea ENTER" << endl; //cout
            cout << boundArea << " " << realArea << endl;

            //Second Filter of bounds for i_img < 3
            if(i_img < 3 && boundArea >= 3000){
                enterBuffer = 0;
                selectedRect = boundRect[i];
                buffer_rect.push_back(selectedRect);

                cout << "bufer_rect.size()" << endl; //cout
                cout << buffer_rect.size() << endl;
                rectangle(img_real, selectedRect.tl(), selectedRect.br(),color, 2);
            }
            else if (i_img >= 3){
                cout << "IOU2RECTS" << endl;
                cout << IOU2Rects(boundRect[i],buffer_rect[i_img-1]) << endl;
                rectangle(img_real,boundRect[i].tl(),boundRect[i].br(),Scalar(255,255,255),2);

                cout << "###if boundRect[i] is sufficient good, then select it###" << endl;
                cout << IOU2Rects(boundRect[i],buffer_rect[i_img-1]) << endl;
                if(IOU2Rects(boundRect[i],buffer_rect[i_img-1]) > 0.5){
                    enterBuffer = 0;
                    cout << "" << endl;
                    cout << "========ENTER BOUND SELECTED RECT LISTO=========" << endl;

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

            Rect actualRect = selectActualRect(selectedRect,boundRect,buffer_rect,IOUs,posBound,i_img,pointsMahalanobis);
            double minArea = abs(buffer_rect[i_img-1].area() - actualRect.area());

            cout << "minArea" << endl; //cout
            cout << abs(buffer_rect[i_img-1].area() - actualRect.area()) << endl;

            //elegir como condicion comparacion ious entre promedio 3 anteriores
            double compareIOU4Rects = IOU4Rects(actualRect,buffer_rect,i_img);
            double compareIOU2Rects = IOU2Rects(actualRect,buffer_rect[i_img-1]);

            selectedRect = selectMethod(actualRect,buffer_rect,i_img,meanPtTl,meanPtBr,compareIOU4Rects,compareIOU2Rects,minArea,enterBuffer,pointsMahalanobis);
        }
        /**
         * Here begins the Kalman filter approach taking the centerActual and the centerPrev
         * 
         * @var Point2d centerActual    the actual center of the selectedRect
         * @var Point2d centerPrev      the previous center of the previous selectedRect (buffer_rect[i_img-1])
         * 
         * @see kalmanTracking
        */
        
        //Point2d centerActual(buffer_rect[i_img].tl().x+abs(buffer_rect[i_img].tl().x - buffer_rect[i_img].br().x)/2,buffer_rect[i_img].tl().y+abs(buffer_rect[i_img].tl().y - buffer_rect[i_img].br().y)/2);
        Point2d centerActual(selectedRect.tl().x+abs(selectedRect.tl().x - selectedRect.br().x)/2,selectedRect.tl().y+abs(selectedRect.tl().y - selectedRect.br().y)/2);
        Point2d centerPrev(buffer_rect[i_img-1].tl().x+abs(buffer_rect[i_img-1].tl().x - buffer_rect[i_img-1].br().x)/2,buffer_rect[i_img-1].tl().y+abs(buffer_rect[i_img-1].tl().y - buffer_rect[i_img-1].br().y)/2);
        Point measPt;
        Point statePt;
        Point predictPt;
        kalmanTracking(centerActual, centerPrev, measPt, statePt, predictPt, mousev, kalmanv, KF);
        //kalmanTracking(centerPrev,centerActual, measPt, statePt, predictPt, mousev, kalmanv, KF);
        //cout << "POINTSSSSSS" << endl;
        //cout << measPt.x << " " << measPt.y << endl;
        //cout << statePt.x << " " << statePt.y << endl;
        //cout << predictPt.x << " " << predictPt.y << endl;
        #define drawCross( center, color, d )                                           \
            line( img_real, Point( center.x - d, center.y - d ),                         \
                            Point( center.x + d, center.y + d ), color, 1, LINE_AA, 0); \
            line( img_real, Point( center.x + d, center.y - d ),                         \
                            Point( center.x - d, center.y + d ), color, 1, LINE_AA, 0 )

        vector<Point2f> pointPrev;
        pointPrev.push_back(centerPrev);
        vector<Point2f> pointCalc = calcOpticalFlow2points(imgPrev,imgActual,pointPrev);
        drawCross( pointCalc[0], Scalar(255,0,255), 5); //morado

        vector<Point2f> pointsVel;
        Point2f vel(centerActual.x-centerPrev.x,centerActual.y-centerPrev.y);
        cout << "###VELOCIDAD ###" << endl;
        cout << norm(vel.x-vel.y) << endl;
        
        //if (norm(centerActual-centerPrev) > 60 ){
        //if (norm(centerActual-centerPrev) > 40 ){
        if (norm(vel.x-vel.y) > 40 && norm(vel.x-vel.y) < 80 ){ // y viene de un motion

        }
        if (norm(vel.x-vel.y) > 40 && norm(vel.x-vel.y) < 80 ){ // y viene de un motion
            //hacer intercambio con promedio
            cout << "===CHANGE BY POINTS===" << endl;
            //Point2d newTl(selectedRect.tl().x - (centerActual.x - centerPrev.x),selectedRect.tl().y - (centerActual.y - centerPrev.y));
            //Point2d newBr(selectedRect.br().x - (centerActual.x - centerPrev.x),selectedRect.br().y -(centerActual.y - centerPrev.y));
            Point2d newTl(selectedRect.tl().x - (centerActual.x - pointCalc[0].x),selectedRect.tl().y - (centerActual.y - pointCalc[0].y));
            Point2d newBr(selectedRect.br().x - (centerActual.x - pointCalc[0].x),selectedRect.br().y -(centerActual.y - pointCalc[0].y));
            selectedRect = Rect(newTl,newBr);
        }
        
        cout << "NORMAAA" <<endl;
        cout << norm(centerActual-centerPrev) << endl;
        buffer_rect.push_back(selectedRect);
        rectangle(img_real, selectedRect.tl(), selectedRect.br(),color, 2);
        //circle(drawing,statePt,2,Scalar(255,255,255));
        //circle(drawing,measPt,2,Scalar(0,0,255));
        //circle(drawing,predictPt,2,Scalar(0,255,0));
        
        circle(img_real, statePt,5, Scalar(255,255,0)); //celeste
        circle(img_real, measPt,5, Scalar(0,255,0)); //verde
        circle(img_real, predictPt,5, Scalar(255,0,0)); //azul
        
        //drawCross( statePt, Scalar(0,0,0), 5 );
        //drawCross( measPt, Scalar(0,255,0), 5 );
        //drawCross( predictPt, Scalar(255,0,0), 5);

        drawCross( centerPrev, Scalar(255,255,255), 5); //blanco
        drawCross( centerActual, Scalar(0,0,255), 5); // rojo
        
        /*
        for (int i = 0; i < mousev.size()-1; i++){
            line(drawing, mousev[i], mousev[i+1], Scalar(255,255,0), 1);
            //cout << "wat" << endl;
        }
        
        for (int i = 0; i < kalmanv.size()-1; i++){
            line(drawing, kalmanv[i], kalmanv[i+1], Scalar(0,155,255), 1);
            //cout << "wat1" << endl;
        }
        */
        //drawCross( statePt, Scalar(255,255,255), 3 );
        //drawCross( measPt, Scalar(0,0,255), 3 );
        //drawCross( predictPt, Scalar(0,255,0), 3 );
        //line( img, statePt, measPt, Scalar(0,0,255), 3, LINE_AA, 0 );
        //line( img, statePt, predictPt, Scalar(0,255,255), 3, LINE_AA, 0 );
    }
    //Mat result;
    //addWeighted(img_real, 0.5, drawing, 0.5, 0.0, result);
    imshow("over",img_real);
    waitKey(0);

    return img_real;
}



void readDisparityImages(String path_in,String path_out,String path_real,double threshMahalanobis){
    vector<cv::String> img_name;
    glob(path_in + "*.pgm", img_name, false);  

    int n_imgs = img_name.size(); //number of png files in images folder

    vector<Rect> buffer_rect;
    KalmanFilter KF(4, 2, 0);
    vector<Point> mousev,kalmanv;

    Mat imgPrev;
    vector<Point2f> pointsPrev;
    int enterBuffer = 0;

    for(int i = 0; i < n_imgs; ++i) {
        String img_real_name = path_real + img_name[i].substr(path_in.length(),img_name[i].length() - path_in.length() -4) + ".jpg";

        cout << img_name[i] << endl;  
        cout << img_real_name << endl;  
        
        Mat img = imread(img_name[i]);
        Mat img_real = imread(img_real_name);
        Mat imgActual = imread(img_real_name,CV_LOAD_IMAGE_GRAYSCALE);

    
        vector<Point2f> pointsActual = detectAndComputePoints(imgActual,false);
        vector<uchar> featuresFound;
        vector<float> err;
        vector<Point2f> pointsCalculated;
        

        filterDisparity(img);
        cout << "filter?" << endl;
        Mat blob;

        vector<Point2f> pointsMahalanobis;

        if(i >0){
            pointsMahalanobis = egoMotion(imgPrev,imgActual,pointsPrev,pointsCalculated,featuresFound,err,threshMahalanobis);
            vector<KeyPoint> kpMahalanobis;
            Mat img_keypoints_mahalanobis;

            KeyPoint::convert(pointsMahalanobis, kpMahalanobis, 10, 1, 0, -1);
            drawKeypoints(imgActual, kpMahalanobis, img_keypoints_mahalanobis, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
            imshow("img_keypoints_mahalanobis", img_keypoints_mahalanobis);
            waitKey(0);
        }
        blob = blobSelection(img,img_real,buffer_rect,i,KF,mousev,kalmanv,enterBuffer,pointsMahalanobis,imgPrev,imgActual);
        std::cout << "blob?" << endl;

        String img_out_name = img_name[i].substr(path_in.length(),img_name[i].length()-path_in.length()-4) + ".jpg";
        String path_save_out = path_out + img_out_name;
        imwrite(path_save_out,blob);



        imgPrev = imgActual;
        pointsPrev = pointsActual;
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
    
    //String path_in = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/IMAGES MIX/images_disp/images_pgm/left/";
    
    //String path_real = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/IMAGES MIX/images_real/left/";
    /*
    String path_in = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/CELLMIX/images_disp/images_pgm/left/";
    String path_out = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/CELLMIX/images_segmentation/left/";
    String path_real = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/CELLMIX/images_real/left/";
    
    */
    String path_in = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/IMAGES MIX/images_disp/images_pgm/left/";
    String path_out = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/IMAGES MIX/images_segmentation/left/";
    String path_real = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/IMAGES MIX/images_real/left/";
    

    double threshMahalanobis = 0.75; // BOOK
    //double threshMahalanobis = 0.4;   // CELLPHONE
    readDisparityImages(path_in,path_out,path_real, threshMahalanobis);

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
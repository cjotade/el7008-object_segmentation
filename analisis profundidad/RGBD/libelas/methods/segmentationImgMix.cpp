/**
 * Segments an object given its disparity image.
 * 
 * @author Camilo Jara Do Nascimento
 */

#define _DEBUG

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

/**
 * Calculates the statics (mu,sigma) given a vector of points
 */
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

/**
 * Select the position of pointsVel that pass the test.
 */
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

/**
 * Do the Ego Motion
 */
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

/**
 * Compute the points using ORB or Shi-Tomasi
 */
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

/**
 * Buffer between given image and the previous three
 */
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

/**
 * Buffer between the previous three images
 */
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

/**
 * Buffer of the actual and the previous image
 */
void buffer2imgs(Rect actualRect,vector<Rect> buffer_rect, int i_img, Point2d& meanPtTl, Point2d& meanPtBr){
    vector<double> w;
    double w0 = 0.6;
    double wActual = 0.4;
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

/**
 * Intersection over Union between actual and the previous Rect
 */
double IOU2Rects(Rect iboundingRectActual, Rect boundingRectPrev){
    Rect interRect = iboundingRectActual & boundingRectPrev;
    Rect unionRect = iboundingRectActual | boundingRectPrev;
	double iou = double(interRect.area()) / unionRect.area();
	return iou;
}

/**
 * Intersection over Union between actual and the previous three Rects
 */
double IOU4Rects(Rect iboundingRectActual, vector<Rect>& buffer_rect, int i_img){
    Rect interRect = iboundingRectActual & buffer_rect[i_img-3] & buffer_rect[i_img-2] & buffer_rect[i_img-1];
    Rect unionRect = iboundingRectActual | buffer_rect[i_img-3] | buffer_rect[i_img-2] | buffer_rect[i_img-1];
	double iou = double(interRect.area()) / unionRect.area();
	return iou;
}

/**
 * Filter the disparity image
 */
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

/**
 * Kalman Tracking given center
 */
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

    //cout << "boundRect.size()" << endl; //cout
    //cout << boundRect.size() << endl;

    for(size_t i = 0; i < contours.size(); i++){
        approxPolyDP(contours[i], contours_poly[i], 3, true); 
        boundRect[i] = boundingRect(contours_poly[i]);
    }
    return boundRect;
}

/**
 * Selects if the actualRect is a good option, else
 * Selects between buffer3 or buffer4
 */ 
Rect selectMethod(Rect actualRect,vector<Rect>& buffer_rect, int i_img,Point2d meanPtTl, Point2d meanPtBr, double compareIOU4Rects,double compareIOU2Rects, double minArea, int& enterBuffer, vector<Point2f> pointsMahalanobis,int threshActualRectArea,int n_buffers,double threshCompIOU2Rects,double threshCompIOU4Rects,bool testMode){
    if(testMode){
        cout << "selectMETHOD " << endl;
        cout << "compareIOU4Rects" << compareIOU4Rects << endl;
        cout << "compareIOU2Rects" << compareIOU2Rects << endl;
        cout << "ACTUAL AREA" << actualRect.area() << endl;
        cout << "ANTERIOR AREA" << buffer_rect[i_img-1].area() << endl;
    }
    //if((compareIOU4Rects >= 0.3 || compareIOU2Rects >= 0.43) && minArea < 20000){ //and areas
    if((compareIOU4Rects >= 0.19 || compareIOU2Rects >= 0.23) && minArea < 18500 && actualRect.area() > 8000){ //and areas minArea > 8000
        enterBuffer = 0;
        if(testMode){
            cout << "enterBuffer COMPARE IOUS" << endl;
            cout << enterBuffer << endl;
            cout << "" << endl;
            cout << "======actualRect ES SUFICIENTEMENTE BUENO====" << endl;
        }
        return actualRect;
    }
    //else if((compareIOU4Rects >= 0.08 || compareIOU2Rects >= 0.09) && minArea < 30000 && actualRect.area() > 14000){
    else if((compareIOU4Rects >= threshCompIOU4Rects || compareIOU2Rects >= threshCompIOU2Rects) && minArea < 30000 && actualRect.area() > threshActualRectArea){
        if(testMode){
            cout << "======actualRect ES SUFICIENTEMENTE BUENO PARTE 2====" << endl;
        }
        enterBuffer = 0;
        return actualRect;
    }
    else{
        if(testMode){
            cout << " " << endl;
            cout << "actualRect.area()" << endl;
            cout << actualRect.area() << endl;
            cout << "buffer_rect[i_img-1].area()" << endl;
            cout << buffer_rect[i_img-1].area() << endl;
        }
        
        Rect selectedRect;
        
        //if(buffer_rect[i_img-1].area() > actualRect.area() && actualRect.area() > 1800  && enterBuffer < 4){
        if(buffer_rect[i_img-1].area() > actualRect.area()  && enterBuffer < n_buffers){
            enterBuffer += 1;
            if(testMode){
                cout << "enterBuffer" << endl;
                cout << enterBuffer << endl;
                cout << "" << endl;
                cout << "BUFFER4" << endl;
            }
            buffer4imgs(actualRect,buffer_rect,i_img,meanPtTl,meanPtBr);
            selectedRect = Rect(meanPtTl,meanPtBr);
        }
        else if (buffer_rect[i_img-1].area() <= actualRect.area() && enterBuffer < n_buffers){
            enterBuffer += 1;
            if(testMode){
                cout << "enterBuffer" << endl;
                cout << enterBuffer << endl;
                cout << "" << endl;
                cout << "BUFFER3" << endl;
            }
            buffer3imgs(buffer_rect,i_img,meanPtTl,meanPtBr);
            selectedRect = Rect(meanPtTl,meanPtBr);
        }
        else{
            enterBuffer = 0;
            if(testMode){
                cout << "enterBuffer" << endl;
                cout << enterBuffer << endl;
                
                cout << "" << endl;
                cout << "==============MOTION==============" << endl;
            }
            //Rect pointsRect = boundingRect(pointsMahalanobis);
            //buffer2imgs(pointsRect,buffer_rect,i_img,meanPtTl,meanPtBr);
            //selectedRect = Rect(meanPtTl,meanPtBr);
            selectedRect = boundingRect(pointsMahalanobis);
        }
        return selectedRect;      
        
    }
}

/**
 * Selects the actual Rect
 */
Rect selectActualRect(Rect selectedRect, vector<Rect> boundRect,vector<Rect>& buffer_rect, vector<double> IOUs,vector<int> posBound, int i_img, vector<Point2f> pointsMahalanobis,int& selByPrev, bool testMode){
    if(testMode){
        cout << "ENTER SELECTED AREA" << endl;
    }
    Point2d meanPtTl;
    Point2d meanPtBr;
    Rect actualRect;
    if(IOUs.size() != 0){ // and maxIOU
        int posMaxIOU = std::distance(IOUs.begin(), std::max_element(IOUs.begin(), IOUs.end()));
        double maxIOU = *max_element(IOUs.begin(), IOUs.end());
        if(testMode){
            cout << "Seleccion por MAXIOU" << endl; //cout
            cout << maxIOU << endl;
        }
        actualRect = boundRect[posBound[posMaxIOU]];
    }
    else{
        if(testMode){
            cout << "Seleccion por imagen anterior" << endl;
            cout << "=============================================================================" << endl;
        }
        //actualRect = buffer_rect[i_img-1]; // VERIFICAR
        selByPrev = 1;
        Rect pointsRect = boundingRect(pointsMahalanobis);
        Point2d mPtTl;
        Point2d mPtBr;
        buffer2imgs(pointsRect,buffer_rect,i_img,mPtTl,mPtBr);
        actualRect = Rect(mPtTl,mPtBr);
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
Mat blobSelection(Mat& img, Mat img_real,vector<Rect>& buffer_rect, int i_img, KalmanFilter KF,vector<Point>& mousev,vector<Point>& kalmanv, int& enterBuffer,vector<Point2f> pointsMahalanobis, Mat imgPrev,Mat imgActual,int threshActualRectArea,int n_buffers,double threshCompIOU2Rects,double threshCompIOU4Rects,bool testMode){
    /**
     * Some variables you need to understand
     * 
     * @var vector<Rect> boundRect     contain all found rois for frame i_img
     * @var vector<double> IOUs        contain the IOUs between the boundRect[i] and the selected Rect in the previous frame (i_img-1)
     * @var vector<int> posBound       contain the position of i for post-calculus of the max IOU in boundRect
     * @var Rect selectedRect          the selected Rect for frame i_img
    */
    if(testMode){
        cout << "image number" << endl;
        cout << i_img+1 << endl;
        cout << "img.size" << endl;
        cout << img.size << endl;
    }
    int selByPrev = 0;

    cv::cvtColor(img, img, CV_RGB2GRAY);
    threshold(img,img, 200, 255, THRESH_BINARY_INV);
    
    RNG rng(12345);
    vector<vector<Point> > contours;
    Scalar color;
    vector<double> IOUs;
    vector<int> posBound;
    Rect selectedRect;

    vector<Rect> boundRect = findContours(img,contours);

    if(testMode){
        cout << "contours.size()" << endl; //cout
        cout << contours.size() << endl;

        cout << " " << endl;
        cout << "SELECCION MEJOR BOUND" << endl;
    }
    for(size_t i = 0; i< contours.size(); i++){
        if(testMode){
            cout << " "<< endl;
            cout << i << "COUNTOUR NUMBER" << endl;
        }
        color = Scalar(rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256));
        int boundArea = boundRect[i].area();
        int realArea = img_real.rows * img_real.cols;

        //First Filter of bounds
        if(boundArea != realArea && boundArea >= 1000){
            if(testMode){
                cout << "boundArea realArea ENTER" << endl; //cout
                cout << boundArea << " " << realArea << endl;
            }

            //Second Filter of bounds for i_img < 3
            if(i_img < 3 && boundArea >= 3000){
                enterBuffer = 0;
                selectedRect = boundRect[i];
                buffer_rect.push_back(selectedRect);

                if(testMode){
                    cout << "bufer_rect.size()" << endl; //cout
                    cout << buffer_rect.size() << endl;
                }

                rectangle(img_real, selectedRect.tl(), selectedRect.br(),color, 2);
            }
            else if (i_img >= 3){
                if(testMode){
                    cout << "IOU2RECTS" << endl;
                    cout << IOU2Rects(boundRect[i],buffer_rect[i_img-1]) << endl;

                    cout << " "<< endl;
                    cout << i << "DIBUJANDO NUMBER" << endl;
                
                    rectangle(img_real,boundRect[i].tl(),boundRect[i].br(),Scalar(255,255,255),2);
               
                    cout << "###if boundRect[i] is sufficient good, then select it###" << endl;
                    cout << IOU2Rects(boundRect[i],buffer_rect[i_img-1]) << endl;
                }
                if(IOU2Rects(boundRect[i],buffer_rect[i_img-1]) > 0.5){
                    enterBuffer = 0;
                    if(testMode){
                        cout << "" << endl;
                        cout << "========ENTER BOUND SELECTED RECT LISTO=========" << endl;
                    }

                    selectedRect = boundRect[i];

                    if(testMode){
                        cout << selectedRect.area() << endl;
                    }
                }
                //dont't select boundRect[i]
                else{ 
                    if(testMode){
                        cout << "" << endl;
                        cout << "ENTEr JUST IN IOUs" << endl;
                    }
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
    if(testMode){
        cout << "" << endl;
        cout << "SELECCION ACTUAL O BUFFERS" << endl;
        cout << "IOUs.size()" << endl;
        cout << IOUs.size() << endl;
    }

    if(i_img >= 3){
        if(testMode){
            cout << "selectedRect.area()" << endl;
            cout << selectedRect.area() << endl;
        }
        
        /**
         * selectedRect.area() != 0 means that boundRect[i] pass the IOU above filter,
         * so is no need to re-select it by the buffers or when by the maximum IOU 
         */
        if(selectedRect.area() == 0){
            Point2d meanPtTl;
            Point2d meanPtBr;

            Rect actualRect = selectActualRect(selectedRect,boundRect,buffer_rect,IOUs,posBound,i_img,pointsMahalanobis,selByPrev,testMode);
            double minArea = abs(buffer_rect[i_img-1].area() - actualRect.area());

            if(testMode){
                cout << "minArea" << endl; //cout
                cout << abs(buffer_rect[i_img-1].area() - actualRect.area()) << endl;
            }

            //elegir como condicion comparacion ious entre promedio 3 anteriores
            double compareIOU4Rects = IOU4Rects(actualRect,buffer_rect,i_img);
            double compareIOU2Rects = IOU2Rects(actualRect,buffer_rect[i_img-1]);

            selectedRect = selectMethod(actualRect,buffer_rect,i_img,meanPtTl,meanPtBr,compareIOU4Rects,compareIOU2Rects,minArea,enterBuffer,pointsMahalanobis,threshActualRectArea,n_buffers,threshCompIOU2Rects,threshCompIOU4Rects,testMode);
        }
        /**
         * Here begins the Kalman filter approach taking the centerActual and the centerPrev
         * 
         * @var Point2d centerActual    the actual center of the selectedRect
         * @var Point2d centerPrev      the previous center of the previous selectedRect (buffer_rect[i_img-1])
         * 
         * @see kalmanTracking
        */
        
        Point2d centerActual(selectedRect.tl().x+abs(selectedRect.tl().x - selectedRect.br().x)/2,selectedRect.tl().y+abs(selectedRect.tl().y - selectedRect.br().y)/2);
        Point2d centerPrev(buffer_rect[i_img-1].tl().x+abs(buffer_rect[i_img-1].tl().x - buffer_rect[i_img-1].br().x)/2,buffer_rect[i_img-1].tl().y+abs(buffer_rect[i_img-1].tl().y - buffer_rect[i_img-1].br().y)/2);
        Point measPt;
        Point statePt;
        Point predictPt;
        kalmanTracking(centerActual, centerPrev, measPt, statePt, predictPt, mousev, kalmanv, KF);
        
        #define drawCross( center, color, d )                                           \
            line( img_real, Point( center.x - d, center.y - d ),                         \
                            Point( center.x + d, center.y + d ), color, 1, LINE_AA, 0); \
            line( img_real, Point( center.x + d, center.y - d ),                         \
                            Point( center.x - d, center.y + d ), color, 1, LINE_AA, 0 )

        vector<Point2f> pointPrev;
        pointPrev.push_back(centerPrev);
        vector<Point2f> pointCalc = calcOpticalFlow2points(imgPrev,imgActual,pointPrev);
        if(testMode){
            drawCross( pointCalc[0], Scalar(255,0,255), 5); //morado
        }

        vector<Point2f> pointsVel;
        Point2f vel(centerActual.x-centerPrev.x,centerActual.y-centerPrev.y);

        if(testMode){
            cout << "###VELOCIDAD ###" << endl;
            cout << norm(vel.x-vel.y) << endl;
        }
        
        //if (norm(centerActual-centerPrev) > 60 ){
        //if (norm(centerActual-centerPrev) > 40 ){
            

        if (selByPrev == 1){ // y viene de un motion
            //hacer intercambio con promedio
            if(testMode){
                cout << "===CHANGE BY POINTS===" << endl;
            }
            Point2d newTl(selectedRect.tl().x - (centerActual.x - pointCalc[0].x),selectedRect.tl().y - (centerActual.y - pointCalc[0].y));
            Point2d newBr(selectedRect.br().x - (centerActual.x - pointCalc[0].x),selectedRect.br().y -(centerActual.y - pointCalc[0].y));
            selectedRect = Rect(newTl,newBr);
            selByPrev = 0;
        }

        if(testMode){
            cout << "NORMAAA" <<endl;
            cout << norm(centerActual-centerPrev) << endl;
        }

        buffer_rect.push_back(selectedRect);
        rectangle(img_real, selectedRect.tl(), selectedRect.br(),color, 2);

        if(testMode){
            circle(img_real, statePt,5, Scalar(255,255,0)); //celeste
            circle(img_real, measPt,5, Scalar(0,255,0)); //verde
            circle(img_real, predictPt,5, Scalar(255,0,0)); //azul
            
            //drawCross( statePt, Scalar(0,0,0), 5 );
            //drawCross( measPt, Scalar(0,255,0), 5 );
            //drawCross( predictPt, Scalar(255,0,0), 5);

            drawCross( centerPrev, Scalar(255,255,255), 5); //blanco
            drawCross( centerActual, Scalar(0,0,255), 5); // rojo
        }
        
    }
    if(testMode){
        imshow("over",img_real);
        waitKey(0);
    }

    return img_real;
}


/**
 * Read disparity Images and relegate to another methods
 */
void readDisparityImages(String path_in,String path_out,String path_real,double threshMahalanobis,int threshActualRectArea,int n_buffers,double threshCompIOU2Rects,double threshCompIOU4Rects,bool testMode){
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

        cout << "DISPARITY IMAGE NAME:" << endl;
        cout << img_name[i] << endl;  
        cout << "REAL IMAGE NAME:" << endl;
        cout << img_real_name << endl;  
        
        Mat img = imread(img_name[i]);
        Mat img_real = imread(img_real_name);
        Mat imgActual = imread(img_real_name,CV_LOAD_IMAGE_GRAYSCALE);

    
        vector<Point2f> pointsActual = detectAndComputePoints(imgActual,false);
        vector<uchar> featuresFound;
        vector<float> err;
        vector<Point2f> pointsCalculated;
        

        filterDisparity(img);
        Mat blob;

        vector<Point2f> pointsMahalanobis;

        if(i >0){
            pointsMahalanobis = egoMotion(imgPrev,imgActual,pointsPrev,pointsCalculated,featuresFound,err,threshMahalanobis);
            vector<KeyPoint> kpMahalanobis;
            Mat img_keypoints_mahalanobis;

            KeyPoint::convert(pointsMahalanobis, kpMahalanobis, 10, 1, 0, -1);
            drawKeypoints(imgActual, kpMahalanobis, img_keypoints_mahalanobis, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
            if(testMode){
                imshow("img_keypoints_mahalanobis", img_keypoints_mahalanobis);
                waitKey(0);
            }
        }
        blob = blobSelection(img,img_real,buffer_rect,i,KF,mousev,kalmanv,enterBuffer,pointsMahalanobis,imgPrev,imgActual, threshActualRectArea, n_buffers,threshCompIOU2Rects,threshCompIOU4Rects,testMode);

        String img_out_name = img_name[i].substr(path_in.length(),img_name[i].length()-path_in.length()-4) + ".jpg";
        String path_save_out = path_out + img_out_name;

        cout << "SAVING IMAGE IN:" << endl;
        cout << path_save_out << endl;
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


    // ========== NO SELECTION ============
    String path_in = "../example/images_disp/left/";
    String path_out = "../example/images_segmentation/";
    String path_real = "../example/images_real/left/";

    // ========== SELECTION ============
    //String path_in = "../example/images_select_disp/left/";
    //String path_out = "../example/images_segmentation_select_disp/";
    //String path_real = "../example/images_real/left/";
    

    bool testMode = false;
    double threshMahalanobis = 0.75; // BOOK
    //double threshMahalanobis = 0.5;   // CELLPHONE NECESITA MAS BUFFERS 
    int n_buffers = 4;  // BOOK=4, CELLPHONE=20
    int threshActualRectArea = 14000; // BOOK=14000 CELLPHONE=6000
    double threshCompIOU4Rects = 0.08; //BOOK = (compareIOU4Rects >= 0.08 || compareIOU2Rects >= 0.09)
    double threshCompIOU2Rects = 0.09; //CELLPHONE = (compareIOU4Rects >= 0.03 || compareIOU2Rects >= 0.05)
    
    readDisparityImages(path_in,path_out,path_real, threshMahalanobis,threshActualRectArea,n_buffers,threshCompIOU2Rects,threshCompIOU4Rects,testMode);

    return 0; 
}

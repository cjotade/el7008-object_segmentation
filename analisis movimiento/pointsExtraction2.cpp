/**
 * Segments an object using motion analysis.
 * 
 * @author Camilo Jara Do Nascimento
 */

#define _DEBUG

#include <opencv2/opencv.hpp>

#include "opencv2/core/version.hpp"
#include <iostream>
#include <numeric>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml.hpp>


using namespace std;
using namespace cv;
using namespace cv::ml;

/**
 * Harris Filter
 */
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

/**
 * Get Harris points given a filtered harris image
 */
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

/**
 * Returns the type of a Mat
 */
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

/**
 * Intersection over Union between actual and the previous Rect
 */
double IOU2Rects(Rect pointsRect, Rect iSelectedRect){
    Rect interRect = pointsRect & iSelectedRect;
    Rect unionRect = pointsRect | iSelectedRect;
	double iou = double(interRect.area()) / unionRect.area();
	return iou;
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
 * Separates the statics in order to use it in the 5 regressors.
 */
void separateStatics(Mat mu, Mat cov,Mat& mu_u, Mat& mu_v, Mat& cov_uu, Mat& cov_vv, Mat& cov_uv){
    mu_u.push_back(mu.at<float>(0));
    mu_v.push_back(mu.at<float>(1));
    cov_uu.push_back(cov.at<float>(0,0));
    cov_vv.push_back(cov.at<float>(1,1));
    cov_uv.push_back(cov.at<float>(0,1));
}

/**
 * Concatenates the predicted statics from every regressor in order to create the mu and sigma matrix.
 */
Mat concatStatics(Mat mu_u, Mat mu_v, Mat cov_uu, Mat cov_vv, Mat cov_uv){
    Mat statics;
    Mat concat_mu;
    Mat concat_cov_uu;
    Mat concat_cov_vv;
    hconcat(mu_u,mu_v,concat_mu);
    hconcat(concat_mu,cov_uu,concat_cov_uu);
    hconcat(concat_cov_uu,cov_vv,concat_cov_vv);
    hconcat(concat_cov_vv,cov_uv,statics);   
    return statics;
}

/**
 * Do de Ego Motion
 */
vector<Point2f> egoMotion(Mat imgPrev, Mat imgActual, vector<Point2f> pointsPrev, vector<Point2f> pointsCalculated, vector<uchar> featuresFound,vector<float> err,vector<Ptr<SVM> > svms, double threshMahalanobis){
    vector<Point2f> pointsMahalanobis;

    //calcOpticalFlowPyrLK(imgPrev, imgActual, pointsPrev,pointsCalculated,featuresFound,err,Size(31,31),3,TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS),20,0.03),OPTFLOW_USE_INITIAL_FLOW);
    calcOpticalFlowPyrLK(imgPrev, imgActual, pointsPrev,pointsCalculated,featuresFound,err);
    Mat cov, mu;
    //restar puntos pa obtener velocidades
    /*
    vector<Point2f> pointsVel;
    for(int i = 0 ; i < pointsPrev.size() ; i++){
        //cout << pointsCalculated[i].x-pointsPrev[i].x << endl;
        //cout << pointsCalculated[i].y-pointsPrev[i].y << endl;
        Point2f vel(pointsCalculated[i].x-pointsPrev[i].x,pointsCalculated[i].y-pointsPrev[i].y);
        pointsVel.push_back(vel);
    }
    */
    calcStatics(pointsCalculated,cov,mu);
    //calcStatics(pointsVel,cov,mu);
    
    Mat mu_u; 
    Mat mu_v;
    Mat cov_uu;
    Mat cov_vv;
    Mat cov_uv; 
    separateStatics(mu,cov,mu_u,mu_v,cov_uu,cov_vv,cov_uv);

    Mat statics = concatStatics(mu_u,mu_v,cov_uu,cov_vv,cov_uv);
    /*
    cout << "ego" << endl;
    cout << statics << endl;
    cout << statics.size() << endl;
    cout << statics.rows << endl;
    cout << statics.cols << endl;
    */
    vector<float> responses;

    for(int i = 0; i < statics.cols; i++){
        Mat testDataMat(1, 1, CV_32FC1, statics.at<float>(i));
        float response = svms[i]->predict(testDataMat);
        responses.push_back(response);
    }
    

    Mat predMu(Size(2,1),CV_32FC1);
    Mat predCov(Size(2,2),CV_32FC1);

    predMu.at<float>(0,0) = responses[0];
    predMu.at<float>(0,1) = responses[1];
    predCov.at<float>(0,0) = responses[2];
    predCov.at<float>(1,1) = responses[3];
    predCov.at<float>(0,1) = responses[4];
    predCov.at<float>(1,0) = responses[4];
    
    Mat icov;
    invert(predCov, icov, DECOMP_SVD);
    icov.convertTo(icov,CV_32FC1);

    //vector<int> vi_pos = selectPosPointsMahalanobis(pointsVel,icov,predMu,threshMahalanobis);
    vector<int> vi_pos = selectPosPointsMahalanobis(pointsCalculated,icov,predMu,threshMahalanobis);

    for(int k=0; k<vi_pos.size(); k++){
        pointsMahalanobis.push_back(pointsCalculated[vi_pos[k]]);
    }

    return pointsMahalanobis;
}

/**
 * Do the Ego Motion in the train set.
 */
void egoMotionTrain(Mat imgPrev, Mat imgActual, vector<Point2f> pointsPrev, vector<Point2f> pointsCalculated, vector<uchar> featuresFound,vector<float> err, Mat& mu_u, Mat& mu_v,Mat& cov_uu,Mat& cov_vv,Mat& cov_uv){
    vector<Point2f> pointsMahalanobis;

    calcOpticalFlowPyrLK(imgPrev, imgActual, pointsPrev,pointsCalculated,featuresFound,err);
    Mat cov, mu;
    //restar puntos para obtener velocidades
    /*
    vector<Point2f> pointsVel;
    for(int i = 0 ; i < pointsPrev.size() ; i++){
        //cout << pointsCalculated[i].x-pointsPrev[i].x << endl;
        //cout << pointsCalculated[i].y-pointsPrev[i].y << endl;
        Point2f vel(pointsCalculated[i].x-pointsPrev[i].x,pointsCalculated[i].y-pointsPrev[i].y);
        pointsVel.push_back(vel);
    }
    */
    
    calcStatics(pointsCalculated,cov,mu);
    //calcStatics(pointsVel,cov,mu);
    
    //cout << mu << endl;
    //cout << cov << endl;

    Mat sepMu_u;
    Mat sepMu_v;
    Mat sepCov_uu;
    Mat sepCov_vv;
    Mat sepCov_uv;
    separateStatics(mu,cov,sepMu_u,sepMu_v,sepCov_uu,sepCov_vv,sepCov_uv);
    mu_u.push_back(sepMu_u);
    mu_v.push_back(sepMu_v);
    cov_uu.push_back(sepCov_uu);
    cov_vv.push_back(sepCov_vv);
    cov_uv.push_back(sepCov_uv);
}

/**
 * Find the centroid of a vector of points
 */
Point2f findCentroid(vector<Point2f> pointsMahalanobis){
    double sumx = 0; 
    double sumy = 0; 
    int i = 0;
    for (int j = 0; j < pointsMahalanobis.size(); j++) {
        sumx += (double)pointsMahalanobis[j].x;
        sumy += (double)pointsMahalanobis[j].y;
        i+=1;
    }
    int meanx = sumx/pointsMahalanobis.size();
    int meany = sumy/pointsMahalanobis.size();

    Point2f meanPt(meanx,meany);
    return meanPt;
}

/**
 * Find the contours given a image
 */
vector<Rect> findContours(Mat img, vector<Point2f> pointsMahalanobis,vector<vector<Point> >& contoursSelected){
    

    //imshow("real",img);
    //waitKey(0);
    
    
    int thresh = 120;
    Canny( img, img, thresh, thresh*2, 3 );
    imshow("canny",img);
    waitKey(0);

    RNG rng(12345);
    Scalar color;
    Mat drawing = Mat::zeros(img.size(), CV_8UC3);
    
    vector<vector<Point> > contours;
    findContours(img, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

    vector<vector<Point> > contours_poly(contours.size());
    vector<Rect> boundRect(contours.size());
    

    for(size_t i = 0; i < contours.size(); i++){
        approxPolyDP(contours[i], contours_poly[i], 3, true); 
        boundRect[i] = boundingRect(contours_poly[i]);
    }
    
    vector<Rect> selectedRects;
    for( size_t i = 0; i< contours.size(); i++ ){
        color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );

        Rect selectedRect;
        int boundArea = boundRect[i].area();
        int realArea = img.rows * img.cols;

        if(boundArea != realArea && boundArea >= 1000){
            if(boundArea >=3000){
                selectedRect = boundRect[i];
                selectedRects.push_back(selectedRect);
                contoursSelected.push_back(contours[i]);
            }
        }
    }

    return selectedRects;
}

/**
 * True if a points is in the Rect.
 */
bool pointInRect(Rect rect, Point2f point){
    if(point.x >= rect.tl().x &&  point.x <= rect.br().x){
        if(point.y >= rect.tl().y &&  point.y <= rect.br().y){
            return true;
        }
    }
    return false;
}

/**
 * Mean of a vector of Rects
 */
void meanRects(vector<Rect> Rects, Point2d& meanPtTl, Point2d& meanPtBr){
    double sumTlx = 0; 
    double sumTly = 0;
    double sumBrx = 0; 
    double sumBry = 0;  
    int i = 0;
    for (int j = 0; j < Rects.size(); j++) {
        sumTlx += Rects[j].tl().x;
        sumTly += Rects[j].tl().y;
        sumBrx += Rects[j].br().x;
        sumBry += Rects[j].br().y;
        i+=1;
    }
    int meanTlx = (int)sumTlx/Rects.size();
    int meanTly = (int)sumTly/Rects.size();
    int meanBrx = (int)sumBrx/Rects.size();
    int meanBry = (int)sumBry/Rects.size();
    
    meanPtTl.x = meanTlx;
    meanPtTl.y = meanTly;
    meanPtBr.x = meanBrx;
    meanPtBr.y = meanBry;
}

/**
 * Find the selectedRects.
 */
vector<Rect> findSelectedRects(Rect pointsRect,vector<Point2f> pointsMahalanobis,vector<Rect> selectedRects){
    vector<Rect> filterRects;
    for(int j = 0 ; j < pointsMahalanobis.size(); j++){
        for(int i=0; i < selectedRects.size() ; i++){
            if(pointInRect(selectedRects[i],pointsMahalanobis[j])){
                //cout << "IOU2Rects(pointsRect,selectedRects[i])" << endl;
                //cout << IOU2Rects(pointsRect,selectedRects[i]) << endl;
                if(IOU2Rects(pointsRect,selectedRects[i]) > 0.08 && selectedRects[i].area() < 24000 && selectedRects[i].area() > 4000){
                    filterRects.push_back(selectedRects[i]);
                }
            }
        }
    }
    return filterRects;
}



int main(void){
    String path_in = "/home/camilojd/Universidad/Primavera 2018/EL7008/ENTREGABLES/analisis movimiento/example/Book Input/";
    String path_out = "/home/camilojd/Universidad/Primavera 2018/EL7008/ENTREGABLES/analisis movimiento/example/Book Output/";
   
    double threshMahalanobis = 0.75;
    int numberTrainImages = 51;
    vector<cv::String> img_name;
    glob(path_in + "*.jpg", img_name, false);  

    int n_imgs = img_name.size(); //number of png files in images folder

    Mat imgPrev;
    vector<Point2f> pointsPrev;

    RNG rng(12345);
    Scalar color;

    Mat mu_u;
    Mat mu_v;
    Mat cov_uv;
    Mat cov_vv;
    Mat cov_uu;

    vector<Ptr<SVM> > svms;
    
    Ptr<SVM> svm_mu_u = SVM::create();
    svm_mu_u->setType(SVM::NU_SVR);
    svm_mu_u->setKernel(SVM::RBF);
    svm_mu_u->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm_mu_u->setNu(0.1);

    Ptr<SVM> svm_mu_v = SVM::create();
    svm_mu_v->setType(SVM::NU_SVR);
    svm_mu_v->setKernel(SVM::RBF);
    svm_mu_v->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm_mu_v->setNu(0.1);

    Ptr<SVM> svm_cov_uu = SVM::create();
    svm_cov_uu->setType(SVM::NU_SVR);
    svm_cov_uu->setKernel(SVM::RBF);
    svm_cov_uu->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm_cov_uu->setNu(0.1);

    Ptr<SVM> svm_cov_vv = SVM::create();
    svm_cov_vv->setType(SVM::NU_SVR);
    svm_cov_vv->setKernel(SVM::RBF);
    svm_cov_vv->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm_cov_vv->setNu(0.1);

    Ptr<SVM> svm_cov_uv = SVM::create();
    svm_cov_uv->setType(SVM::NU_SVR);
    svm_cov_uv->setKernel(SVM::RBF);
    svm_cov_uv->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm_cov_uv->setNu(0.1);

    for(int i = 0; i < n_imgs; ++i) {
        Mat imgActual = imread(img_name[i],CV_LOAD_IMAGE_GRAYSCALE);
        Mat imgReal = imread(img_name[i]);

        cout << "IMAGE NAME:" << endl;
        cout << img_name[i] << endl;  
        
        vector<Point2f> pointsActual = detectAndComputePoints(imgActual,false);
    
        vector<uchar> featuresFound;
        vector<float> err;
        vector<Point2f> pointsCalculated;
        
        if(i > 0){
            if(i < numberTrainImages){
                egoMotionTrain(imgPrev,imgActual,pointsPrev,pointsCalculated,featuresFound,err,mu_u,mu_v,cov_uu,cov_vv,cov_uv);
            }
            else{
                if(i == numberTrainImages){
                    Mat statics = concatStatics(mu_u, mu_v, cov_uu, cov_vv, cov_uv);
                    //TRAIN SVM

                    Mat staticsActual = statics(Rect(0,0,statics.cols,statics.rows-1));
                    Mat staticsFuture = statics(Rect(0,1,statics.cols,statics.rows-1));

                
                    Ptr<TrainData> td_mu_u = TrainData::create(staticsActual(Rect(0,0,1,staticsActual.rows)), ROW_SAMPLE, staticsFuture(Rect(0,0,1,staticsFuture.rows))); 
                    Ptr<TrainData> td_mu_v = TrainData::create(staticsActual(Rect(1,0,1,staticsActual.rows)), ROW_SAMPLE, staticsFuture(Rect(1,0,1,staticsFuture.rows))); 
                    Ptr<TrainData> td_cov_uu = TrainData::create(staticsActual(Rect(2,0,1,staticsActual.rows)), ROW_SAMPLE, staticsFuture(Rect(2,0,1,staticsFuture.rows))); 
                    Ptr<TrainData> td_cov_vv = TrainData::create(staticsActual(Rect(3,0,1,staticsActual.rows)), ROW_SAMPLE, staticsFuture(Rect(3,0,1,staticsFuture.rows))); 
                    Ptr<TrainData> td_cov_uv = TrainData::create(staticsActual(Rect(4,0,1,staticsActual.rows)), ROW_SAMPLE, staticsFuture(Rect(4,0,1,staticsFuture.rows))); 
                
                    svm_mu_u->train(td_mu_u);
                    svm_mu_v->train(td_mu_v);
                    svm_cov_uu->train(td_cov_uu);
                    svm_cov_vv->train(td_cov_vv);
                    svm_cov_uv->train(td_cov_uv); 

                    svms.push_back(svm_mu_u);
                    svms.push_back(svm_mu_v);
                    svms.push_back(svm_cov_uu);
                    svms.push_back(svm_cov_vv);
                    svms.push_back(svm_cov_uv);
                }

                vector<Point2f> pointsMahalanobis = egoMotion(imgPrev,imgActual,pointsPrev,pointsCalculated,featuresFound,err,svms,threshMahalanobis);
            
                vector<KeyPoint> kpMahalanobis;
                vector<KeyPoint> kpReal;
                Mat img_keypoints_mahalanobis;
                Mat img_keypoints_Real;

                KeyPoint::convert(pointsMahalanobis, kpMahalanobis, 10, 1, 0, -1);
                drawKeypoints(imgReal, kpMahalanobis, img_keypoints_mahalanobis, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
                //imshow("img_keypoints_mahalanobis", img_keypoints_mahalanobis);
                //waitKey(0);

                KeyPoint::convert(pointsCalculated, kpReal, 10, 1, 0, -1);
                drawKeypoints(imgActual, kpReal, img_keypoints_Real, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
                //imshow("img_keypoints_Real", img_keypoints_Real);
                //waitKey(0);

                Point2f meanPt = findCentroid(pointsMahalanobis);

                //Mat concat;
                //hconcat(img_keypoints_Real,img_keypoints_mahalanobis,concat);
                //imshow("concat",concat);

                String img_out_name = img_name[i].substr(path_in.length(),img_name[i].length()-path_in.length()-4) + ".jpg";
                String path_save_out = path_out + img_out_name;
                //imwrite(path_save_out,concat);
                
                //Rect pointsRect = findRectByPoints(imgActual,pointsMahalanobis);
                Rect pointsRect = boundingRect(pointsMahalanobis);

                rectangle(img_keypoints_mahalanobis,pointsRect,Scalar(0,255,255));
               
                #define drawCross( center, color, d )                                       \
                line( img_keypoints_mahalanobis, Point( center.x - d, center.y - d ),                         \
                                Point( center.x + d, center.y + d ), color, 1, LINE_AA, 0); \
                line( img_keypoints_mahalanobis, Point( center.x + d, center.y - d ),                         \
                                Point( center.x - d, center.y + d ), color, 1, LINE_AA, 0 )

                drawCross( meanPt, Scalar(255,255,255), 5 );
                
                //imshow("over",img_keypoints_mahalanobis);
                //waitKey(0);

                imwrite(path_save_out,img_keypoints_mahalanobis);
                cout << "SAVING IMAGE IN:" << endl;
                cout << path_save_out << endl;
            }
        
        }   
        imgPrev = imgActual;
        pointsPrev = pointsActual;
    }

    return 0; 
}

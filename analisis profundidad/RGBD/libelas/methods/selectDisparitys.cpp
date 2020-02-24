/**
 * Selects many disparity images given a threshold by its standard desviation.
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
#include <opencv2/ml.hpp>

#include <iostream>
#include <numeric>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace cv::ml;

void readDisparityImages(String path_in,String path_out,int stdThresh){
    vector<cv::String> img_name;
    glob(path_in + "*.pgm", img_name, false);  

    int n_imgs = img_name.size(); //number of png files in images folder


    for(int i = 0; i < n_imgs; ++i) {
        cout << img_name[i] << endl;    
        
        Mat img = imread(img_name[i]);

        //cvtColor(img, img, CV_BGR2HSV);

        int v_bins = 50;
        int histSize[] = { v_bins };
        cv::MatND hist;

        float v_ranges[] = { 0, 255 };
        vector<Mat> channel(3);
        split(img, channel);

        const float* ranges[] = { v_ranges };
        int channels[] = { 0 };

        cv::calcHist(&channel[2], 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false); //histogram calculation

        cv::Scalar mean, stddev;
        cv::meanStdDev(hist, mean, stddev);

        std::cout << "Mean: " << mean[0] << "   StdDev: " << stddev[0] << std::endl;



        String img_out_name = img_name[i].substr(path_in.length(),img_name[i].length()-path_in.length()-4) + ".pgm";
        String path_save_out = path_out + img_out_name;

        if(stddev[0] < stdThresh){
            cvtColor(img,img,CV_BGR2GRAY);
            std::vector<int> compression_params;
            compression_params.push_back(CV_IMWRITE_PXM_BINARY);
            compression_params.push_back(1);
            imwrite(path_save_out,img,compression_params);
        }

        

    }
    
}


int main(void){

    String path_in = "../example/images_disp/left/";
    String path_out = "../example/images_select_disp/left/";
    int stdThresh = 20000;
    readDisparityImages(path_in,path_out,stdThresh);

    return 0;
}

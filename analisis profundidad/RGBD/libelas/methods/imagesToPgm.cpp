/**
 * Transforms an jpg image to pgm image.
 * 
 * @author Camilo Jara Do Nascimento
 */

#define _DEBUG

#include <opencv2/opencv.hpp>

#include "opencv2/core/version.hpp"
#include <iostream>
#include <numeric>
#include <algorithm>

using namespace std;
using namespace cv;


void saveImgAsPgm(String path_in_left,String path_in_right,String path_out_left,String path_out_right){
  vector<cv::String> img_name_left;
  glob(path_in_left + "*.jpg", img_name_left, false);  

  vector<cv::String> img_name_right;
  glob(path_in_right + "*.jpg", img_name_right, false);  

  int n_imgs = img_name_left.size(); //number of png files in images folder

  
  for(int i = 0; i < n_imgs; ++i) {
    Mat img_left = imread(img_name_left[i],CV_LOAD_IMAGE_GRAYSCALE);
    Mat img_right = imread(img_name_right[i],CV_LOAD_IMAGE_GRAYSCALE);
  
    String name_left_pgm = img_name_left[i].substr(path_in_left.length(),img_name_left[i].length() - path_in_left.length() -4) + ".pgm";
    /*
    cout << "img_name i" << endl;
    cout << img_name_left[i] << endl;
    cout << "name_left_pgm" << endl;
    cout << name_left_pgm << endl;
    */
    String name_right_pgm = img_name_right[i].substr(path_in_right.length(),img_name_right[i].length() -path_in_right.length()-4) + ".pgm";
    
    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PXM_BINARY);
    compression_params.push_back(1);
    
    imwrite(path_out_left+name_left_pgm,img_left,compression_params);
    imwrite(path_out_right+name_right_pgm,img_right,compression_params);
  }
}


int main(void){
  String path_in_left = "../example/images_real/left/";
  String path_in_right = "../example/images_real/right/";
  String path_out_left = "../example/images_pgm/left/";
  String path_out_right = "../example/images_pgm/right/";
  saveImgAsPgm(path_in_left,path_in_right,path_out_left,path_out_right);

  return 0; 
}

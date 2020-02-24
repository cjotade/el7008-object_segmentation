/*
Copyright 2011. All rights reserved.
Institute of Measurement and Control Systems
Karlsruhe Institute of Technology, Germany

This file is part of libelas.
Authors: Andreas Geiger

libelas is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or any later version.

libelas is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
libelas; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA 
*/

// Demo program showing how libelas can be used, try "./elas -h" for help


#include <string.h> // memset()
#include <string>

#include <sstream> 
#include "elas.h"
#include "image.h"

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <numeric>
#include <algorithm>

#include <sys/types.h>
#include <dirent.h>

using namespace std;

//using namespace cv;

// compute disparities of pgm image input pair file_1, file_2
void process (const char* file_1,const char* file_2) {

  cout << "Processing: " << file_1 << ", " << file_2 << endl;

  // load images
  image<uchar> *I1,*I2;
  I1 = loadPGM(file_1);
  I2 = loadPGM(file_2);

  // check for correct size
  if (I1->width()<=0 || I1->height() <=0 || I2->width()<=0 || I2->height() <=0 ||
      I1->width()!=I2->width() || I1->height()!=I2->height()) {
    cout << "ERROR: Images must be of same size, but" << endl;
    cout << "       I1: " << I1->width() <<  " x " << I1->height() << 
                 ", I2: " << I2->width() <<  " x " << I2->height() << endl;
    delete I1;
    delete I2;
    return;    
  }

  // get image width and height
  int32_t width  = I1->width();
  int32_t height = I1->height();

  // allocate memory for disparity images
  const int32_t dims[3] = {width,height,width}; // bytes per line = width
  float* D1_data = (float*)malloc(width*height*sizeof(float));
  float* D2_data = (float*)malloc(width*height*sizeof(float));

  // process
  Elas::parameters param;
  param.postprocess_only_left = false;
  Elas elas(param);
  elas.process(I1->data,I2->data,D1_data,D2_data,dims);

  // find maximum disparity for scaling output disparity images to [0..255]
  float disp_max = 0;
  for (int32_t i=0; i<width*height; i++) {
    if (D1_data[i]>disp_max) disp_max = D1_data[i];
    if (D2_data[i]>disp_max) disp_max = D2_data[i];
  }

  // copy float to uchar
  image<uchar> *D1 = new image<uchar>(width,height);
  image<uchar> *D2 = new image<uchar>(width,height);
  for (int32_t i=0; i<width*height; i++) {
    D1->data[i] = (uint8_t)max(255.0*D1_data[i]/disp_max,0.0);
    D2->data[i] = (uint8_t)max(255.0*D2_data[i]/disp_max,0.0);
  }

  // save disparity images
  char output_1[1024];
  char output_2[1024];
  strncpy(output_1,file_1,strlen(file_1)-4);
  strncpy(output_2,file_2,strlen(file_2)-4);
  output_1[strlen(file_1)-4] = '\0';
  output_2[strlen(file_2)-4] = '\0';
  strcat(output_1,"_disp.pgm");
  strcat(output_2,"_disp.pgm");
  savePGM(D1,output_1);
  savePGM(D2,output_2);


  // free memory
  delete I1;
  delete I2;
  delete D1;
  delete D2;
  free(D1_data);
  free(D2_data);
  
}

void processAndOut(const char* file_1,const char* file_2,const char* path_out) {
  cout << "Processing: " << file_1 << ", " << file_2 << endl;

  // load images
  image<uchar> *I1,*I2;
  I1 = loadPGM(file_1);
  I2 = loadPGM(file_2);
  
  // check for correct size
  if (I1->width()<=0 || I1->height() <=0 || I2->width()<=0 || I2->height() <=0 ||
      I1->width()!=I2->width() || I1->height()!=I2->height()) {
    cout << "ERROR: Images must be of same size, but" << endl;
    cout << "       I1: " << I1->width() <<  " x " << I1->height() << 
                 ", I2: " << I2->width() <<  " x " << I2->height() << endl;
    delete I1;
    delete I2;
    return;    
  }

  // get image width and height
  int32_t width  = I1->width();
  int32_t height = I1->height();

  // allocate memory for disparity images
  const int32_t dims[3] = {width,height,width}; // bytes per line = width
  float* D1_data = (float*)malloc(width*height*sizeof(float));
  float* D2_data = (float*)malloc(width*height*sizeof(float));

  // process
  Elas::parameters param;
  param.postprocess_only_left = false;
  Elas elas(param);
  elas.process(I1->data,I2->data,D1_data,D2_data,dims);

  // find maximum disparity for scaling output disparity images to [0..255]
  float disp_max = 0;
  for (int32_t i=0; i<width*height; i++) {
    if (D1_data[i]>disp_max) disp_max = D1_data[i];
    if (D2_data[i]>disp_max) disp_max = D2_data[i];
  }

  // copy float to uchar
  image<uchar> *D1 = new image<uchar>(width,height);
  image<uchar> *D2 = new image<uchar>(width,height);
  for (int32_t i=0; i<width*height; i++) {
    D1->data[i] = (uint8_t)max(255.0*D1_data[i]/disp_max,0.0);
    D2->data[i] = (uint8_t)max(255.0*D2_data[i]/disp_max,0.0);
  }

  
  // save disparity images
  char output_1[1024] = "";
  char output_2[1024] = "";
  char file_left[1024] = "";
  char file_right[1024] = "";
  char out_left[1024] = "";
  char out_right[1024] = "";
  
  strncpy(file_left,file_1,strlen(file_1)-4);
  strncpy(file_right,file_2,strlen(file_2)-4);
  file_left[strlen(file_1)-4] = '\0';
  file_right[strlen(file_2)-4] = '\0';

  strcat(file_left,".pgm");
  strcat(file_right,".pgm");

  strncpy(output_1,path_out,strlen(path_out));
  strncpy(output_2,path_out,strlen(path_out));

  
  // MODIFICAR ESTO DEPENDIENDO NOMBRES IMAGENES
  strncpy (out_left, file_left+strlen(file_left)-17, 17);
  out_left[17] = '\0';
  strncpy (out_right, file_right+strlen(file_right)-18, 18);
  out_right[18] = '\0';
  
  strcat(output_1,out_left);
  strcat(output_2,out_right);

  //strcat(output_1,file_left);
  //strcat(output_2,file_right);
  /*
  cout << "GUARDD" << endl;
  cout << output_1 << endl;
  cout << output_2 << endl;
  cout << "GUARDD" << endl;
  */
  savePGM(D1,output_1);
  savePGM(D2,output_2);

  cout << "Saving image left in: " << output_1 << endl;
  cout << "Saving image right in: " << output_2 << endl;
  
  // free memory
  delete I1;
  delete I2;
  delete D1;
  delete D2;
  free(D1_data);
  free(D2_data);
}

void read_directory(const std::string& name, vector<string>& v)
{
    DIR* dirp = opendir(name.c_str());
    struct dirent * dp;
    int i = 0;
    cout << name.c_str() << endl;
    while ((dp = readdir(dirp)) != NULL) {
      v.push_back(dp->d_name);
      cout << v[i] << endl;
      i+=1;
    }
    closedir(dirp);
    sort(v.begin(), v.end(),std::greater<string>());
}

vector<vector<string> > parseCSV(string path_csv){

    std::ifstream data(path_csv.c_str());
    std::string line;
    vector<vector<string> > parsedCsv;
    while(std::getline(data,line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        vector<string> parsedRow;
        while(std::getline(lineStream,cell,' ')){
            parsedRow.push_back(cell);
            
        }
        parsedCsv.push_back(parsedRow);
    }
    
    return parsedCsv;
}

int main (int argc, char** argv) {

  // run demo
  if (argc==2 && !strcmp(argv[1],"demo")) {
    process("img/cones_left.pgm",   "img/cones_right.pgm");
    process("img/aloe_left.pgm",    "img/aloe_right.pgm");
    process("img/raindeer_left.pgm","img/raindeer_right.pgm");
    process("img/urban1_left.pgm",  "img/urban1_right.pgm");
    process("img/urban2_left.pgm",  "img/urban2_right.pgm");
    process("img/urban3_left.pgm",  "img/urban3_right.pgm");
    process("img/urban4_left.pgm",  "img/urban4_right.pgm");
    cout << "... done!" << endl;

  // compute disparity from input pair
  } else if (argc==3) {
    process(argv[1],argv[2]);
    cout << "... done!" << endl;
    
  } else if (argc==4) {
    vector<string> images_left;
    vector<string> images_right;
    read_directory(argv[1],images_left);
    read_directory(argv[2],images_right);
    
    int n_imgs = std::min(images_left.size(),images_right.size()); //number of png files in images folder
    
    for(int i = 0; i < n_imgs; ++i) {
      char name_img_left[1024] = "";
      char file_left[1024] = "";
      char name_img_right[1024] = "";
      char file_right[1024] = "";  

      if( strcmp(images_left[i].c_str(), ".") != 0 && strcmp(images_left[i].c_str(), "..") != 0 && strcmp(images_right[i].c_str(), ".") != 0 && strcmp(images_right[i].c_str(), "..") != 0 ){
        strcpy(name_img_left, images_left[i].c_str());
        strcpy(file_left,argv[1]);
        strcat(file_left,name_img_left);
        
        strcpy(name_img_right, images_right[i].c_str());
        strcpy(file_right,argv[2]);
        strcat(file_right,name_img_right);
        
        processAndOut(file_left, file_right,argv[3]);
      }
      
    }
    cout << "... done!" << endl;
  
  } else if (argc==5) {
    vector<string> images_left;
    vector<string> images_right;
    read_directory(argv[1],images_left);
    read_directory(argv[2],images_right);

    int n_imgs = std::min(images_left.size(),images_right.size());
    
    vector<vector<string> > parsedCsv = parseCSV(argv[4]);
    for(int i = 0; i < n_imgs; ++i) {
      char name_img_left[1024] = "";
      char file_left[1024] = "";
      char name_img_right[1024] = "";
      char file_right[1024] = ""; 

      string image_left_parsed = parsedCsv[i][5];
      string image_right_parsed = parsedCsv[i][0];
      bool isFileOnDirectoryLeft = (std::find(images_left.begin(), images_left.end(), image_left_parsed) != images_left.end());
      bool isFileOnDirectoryRight = (std::find(images_right.begin(), images_right.end(), image_right_parsed) != images_right.end());
      if(isFileOnDirectoryLeft && isFileOnDirectoryRight){
        
        char image_left[1024];
        char image_right[1024];
        strcpy(image_left, image_left_parsed.c_str());
        strcpy(image_right, image_right_parsed.c_str());

        strcpy(name_img_left, image_left);
        strcpy(file_left,argv[1]);
        strcat(file_left,name_img_left);
        
        strcpy(name_img_right, image_right);
        strcpy(file_right,argv[2]);
        strcat(file_right,name_img_right);
        
        processAndOut(file_left, file_right,argv[3]);
      }
    }
  
    cout << "... done!" << endl;

  // display help
  } else {
    cout << endl;
    cout << "ELAS demo program usage: " << endl;
    cout << "./elas demo ................ process all test images (image dir)" << endl;
    cout << "./elas left.pgm right.pgm .. process a single stereo pair" << endl;
    cout << "./elas -h .................. shows this help" << endl;
    cout << endl;
    cout << "Note: All images must be pgm greylevel images. All output" << endl;
    cout << "      disparities will be scaled such that disp_max = 255." << endl;
    cout << "For more help see README ELAS COMMAND.txt" << endl;
    cout << endl;
  }

  return 0;
}



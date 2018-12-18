cmake_minimum_required (VERSION 2.6)
project (PROYECTO) 
include_directories (include) 
find_package( OpenCV REQUIRED )
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
LINK_DIRECTORIES(${OpenCV_LIBRARY_DIRS})
add_executable (seg segmentationImg.cpp)
target_link_libraries( segmentation ${OpenCV_LIBS} )


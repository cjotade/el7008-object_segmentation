Autor: Camilo Jara Do Nascimento

######################################################################## Package for segmentation task #############################################################################

									    Important Folders:
====================================================================================================================================================================================
src: Methods that calculate the disparity map (more info in README ELAS COMMAND.txt)

methods:
	- imagesToPgm.cpp: transform the images from jpg to pgm (you can also change jpg to another format, but pgm is needed to calculate the disparty maps)
	- selectDisparitys.cpp: select the disparitys images from its standard desviation given a threshold 
	- segmentationImgMix.cpp: segments the objects from a given image.
====================================================================================================================================================================================



									    Important READMES
====================================================================================================================================================================================
1) ./ENTREGABLES/analisis profundidad/RGBD/libelas/README.TXT: use "cmake .", "make" for compile libelas package so then you can elas command (see 3)

====================================================================================================================================================================================
2) ./ENTREGABLES/analisis profundidad/RGBD/libelas/README ELAS COMMAND.txt: the elas commands
In order to run the disparity map calculus just do:

./elas example/images_pgm/left/ example/images_pgm/right/ example/images_disp/ example/images_pgm/img_info_LR.txt

1st arg: path where are the left images in pgm
2nd arg: path where are the right images in pgm
3rd arg: path to save disparity images
4th arg: path where is the images information to find the correspondence between left and right

====================================================================================================================================================================================
3) ./ENTREGABLES/analisis profundidad/RGBD/libelas/methods/README.txt: Explanation of the parameters you can change
Parameters that you need to know and maybe change:


imagesToPgm.cpp:
	- path_in_left: the path of the left images in jpg
	- path_in_right: the path of the right images in jpg
	- path_out_left: the path to save the left images in pgm
	- path_out_right: the path to save the right images in pgm

selectDisparitys.cpp:
	- path_in: the path of the disparity images
	- path_out: the path to save the filtered disparity images
	- stdThresh: the threshold to select disparitys by standard desviation

segmentationImgMix.cpp:
	- path_in: the path of the disparity images
	- path_out: the path to save the segmentation images
	- path_real: the path of jpg images
	- testMode: if true enter in the test mode.
	- threshMahalanobis: the threshold to discard some points in the motion analysis
	- n_buffers: number of max consecutive buffers.
	- threshActualRectArea: threshold in selectMethod used to set the min area of the actualRect
	- threshCompIOU4Rects: threshold used to set the min of the comparison between actualRect and the three previous
	- threshCompIOU2Rects: threshold used to set the min of the comparison between actualRect and the previous one


====================================================================================================================================================================================

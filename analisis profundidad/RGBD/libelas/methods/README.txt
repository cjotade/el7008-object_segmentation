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



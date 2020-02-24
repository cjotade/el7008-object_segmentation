Here is an example:

- images_real: the images in jpg of the left and the right camera (already in folder)

- images_pgm: in order to transform the images_real in jpg into pgm use the imagesToPgm.cpp in methods folder

- images_disp: in order to obtain the disparity images use the elas command in the README ELAS COMMAND.txt (go to source root of libelas)

- images_segmentation: in order to obtain the segmentation image use the segmentationImgMix.cpp in methods folder

- images_select_disp (optional): you can improve the perform task if you select just many disparity images, use selectDisparitys.cpp in methods

- images_segmentation_select_disp: the same as images_segmentation but in this case using the images_select_disp (change the path_out in segmentationImgMix.cpp)

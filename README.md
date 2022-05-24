# Methods for prostate cancer segmentation and detection for multi-parametric MRI

Here I develop and present methods for segmenting prostate cancer from different MRI modalities. I use both ProstateX and PI-CAI (under development).

## Implemented methods for segmentation

* U-Net - different versions are required for 2D and 3D, but here I developed a class that is able to coordinate the operations to setup both (this idea was based on the MONAI implementation of the U-Net)
* Anysotropic Hybrid network (AHNet) - this network is first trained to segment 2D images and some of the (enconding) layers are then transferred to 3D (mostly by either concatenating weights or adding an extra dimension to the layer).

## Implemented methods for detection

*Coming soon! For future reference: I want to do some work with different YOLO versions in both 2D and 3D.*
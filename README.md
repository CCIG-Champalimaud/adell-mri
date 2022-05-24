# Methods for prostate cancer segmentation and detection for multi-parametric MRI

Here I develop and present methods for segmenting prostate cancer from different MRI modalities. I use both ProstateX and PI-CAI (under development).

## Implemented methods for segmentation

* [**U-Net**](https://www.nature.com/articles/s41592-018-0261-2) - different versions are required for 2D and 3D, but here I developed a class that is able to coordinate the operations to setup both (this idea was based on the MONAI implementation of the U-Net)
* [**Anysotropic Hybrid network (AHNet)**](https://arxiv.org/abs/1711.08580) - this network is first trained to segment 2D images and some of the (enconding) layers are then transferred to 3D (mostly by either concatenating weights or adding an extra dimension to the layer).

## Implemented methods for detection

I have only implemented YOLOv2 (YOLO9000) but I have not tested this yet.

*More coming soon! For future reference: I want to do some work with different YOLO versions in both 2D and 3D.*

## Code map

### Modules and networks

I have placed most of the scripts and implementations under `lib/modules/segmentation.py` and `lib/modules/object-detection.py`. `lib/modules/drei` and `lib/modules/zwei` contain building blocks for 3D and 2D neural networks, respectively.

#### Adaptations to PyTorch Lightning

I use PyTorch Lightning to train my models as it offers a very comprehensive set of tools for optimisation. For this reason, in `lib/modules/segmentation_pl.py` I have implemented some classes which inherit from the networks implemented in `lib/modules/segmentation.py` (for now) so that they can be trained using PyTorch Lightning.

### Loss functions

In `lib/modules/losses.py` I have coded most losses necessary based loosely on a paper by [Yeung *et al.*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8785124/) introducing the unified focal loss. For now only segmentation losses are available.

### Data operations

Data functionalities are available in `lib/modules/dataoperations`. This is mostly ways to interact and extend with the MONAI dataset classes.

### Tests

I have included a few unit tests in `lib/tests`. In them, I confirm that networks and modules are outputing the correct shapes, mostly.
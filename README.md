# Methods for prostate cancer segmentation and detection for multi-parametric MRI

Here I develop and present methods for segmenting prostate cancer from different MRI modalities. I use both ProstateX and PI-CAI (under development).

## Implemented methods for segmentation

* [**U-Net**](https://www.nature.com/articles/s41592-018-0261-2) - different versions are required for 2D and 3D, but here I developed a class that is able to coordinate the operations to setup both (this idea was based on the MONAI implementation of the U-Net)
* [**Anysotropic Hybrid network (AHNet)**](https://arxiv.org/abs/1711.08580) - this network is first trained to segment 2D images and some of the (enconding) layers are then transferred to 3D (mostly by either concatenating weights or adding an extra dimension to the layer).

## Implemented methods for detection

* YOLO-based network for 3d detection
* Also implemented a coarse segmentation algorithm, similar to YOLO but outputs only the object probability mask

## Implemented methods for object classification

* ResNet-based methods mostly, but others can be easily constructed with the building blocks in `lib/modules/layers.py`

## Code map

### Modules and networks

I have placed most of the scripts and implementations under `lib/modules/segmentation.py` and `lib/modules/object_detection.py`. `lib/modules/layers.py` contain building blocks for 3D and 2D neural networks.

#### Adaptations to PyTorch Lightning

I use PyTorch Lightning to train my models as it offers a very comprehensive set of tools for optimisation. I.e. in `lib/modules/segmentation_pl.py` I have implemented some classes which inherit from the networks implemented in `lib/modules/segmentation.py` so that they can be trained using PyTorch Lightning.

### Segmentation loss functions

In `lib/modules/losses.py` I have coded most losses necessary based loosely on a paper by [Yeung *et al.*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8785124/) introducing the unified focal loss.

### Tests

I have included a few unit tests in `lib/tests`. In them, I confirm that networks and modules are outputing the correct shapes, mostly.
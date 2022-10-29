# Methods for prostate cancer segmentation and detection for multi-parametric MRI

Here I develop and present methods for segmenting prostate cancer from different MRI modalities. I use both ProstateX and PI-CAI (under development).

## Implemented methods for segmentation

* [**U-Net**](https://www.nature.com/articles/s41592-018-0261-2) - different versions are required for 2D and 3D, but here I developed a class that is able to coordinate the operations to setup both (this idea was based on the MONAI implementation of the U-Net)
* [**U-Net++**](https://pubmed.ncbi.nlm.nih.gov/32613207/) - very similar to U-Net but features [DenseNet](https://arxiv.org/abs/1608.06993)-like skip connections and skip connections between different resolutions. Also features deep supervision at the level of intermediate skip connections
* [**Anysotropic Hybrid network (AHNet)**](https://arxiv.org/abs/1711.08580) - this network is first trained to segment 2D images and some of the (enconding) layers are then transferred to 3D (mostly by either concatenating weights or adding an extra dimension to the layer).
* **Branched input U-Net (BrUNet)** - a U-Net model that has different encoders for each input channel

## Implemented methods for detection

* YOLO-based network for 3d detection
* Also implemented a coarse segmentation algorithm, similar to YOLO but outputs only the object probability mask

## Implemented methods for object classification

* ResNet-based methods mostly, but others can be easily constructed with the building blocks in `lib/modules/layers.py`

## Implemented methods for self-supervised learning

* [**BYOL**](https://arxiv.org/abs/2006.07733) - the paper that proposed a student/teacher type of setup where the teacher is nothing more than a exponential moving average of the whole model
* [**SimSiam**](https://arxiv.org/abs/2011.10566) - the paper that figured out that all you *really* need for self-supervised learning is a stop gradient on one of the encoders
* [**VICReg**](https://arxiv.org/abs/2105.04906) - the paper that figured out that all you *reaaaaally* need for self-supervised learning is a loss function capable of minimising the absence of variance and the covariance of representations and the invariance of pairs of representations for different views on the same image. This framework enables something even better - the networks for the two (or more) views can be wildly different with this loss, so there are **no** constraints on the inputs! So the two "views" can come from distinctly different images paired through some other criteria (in clinical settings this can mean same individual or same disease, for instance)
* [**VICRegL**](https://arxiv.org/abs/2210.01571) - VICReg but works better for segmentation problems. Adds a term which minimises the same as VICReg 

## Code map

### Modules and networks

I have placed most of the scripts and implementations under `lib/modules/segmentation.py`, `lib/modules/classification.py`, `lib/modules/object_detection.py` and `lib/modules/self_supervised.py`. `lib/modules/layers.py` contain building blocks for 3D and 2D neural networks.

#### Adaptations to PyTorch Lightning

I use PyTorch Lightning to train my models as it offers a very comprehensive set of tools for optimisation. I.e. in `lib/modules/segmentation_pl.py` I have implemented some classes which inherit from the networks implemented in `lib/modules/segmentation.py` so that they can be trained using PyTorch Lightning. The same has been done for the self-supervised learning models (in `lib/modules/self_supervised_pl.py`).

### Segmentation loss functions

In `lib/modules/losses.py` I have coded most losses necessary based loosely on a paper by [Yeung *et al.*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8785124/) introducing the unified focal loss.

### Self-supervised loss functions

In `lib/modules/self_supervised.py` you can find all the relevant loss functions.

### Tests

I have included a few unit tests in `testing`. In them, I confirm that networks and modules are outputing the correct shapes and that they are compiling correctly. They are prepared to run with `pytest`, i.e. `pytest testing` runs all of the tests.
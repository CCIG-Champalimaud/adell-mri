# ADeLL-MRI - a Deep-Learning Library for MRI

Here I implement and develop methods for **classification**, **segmentation**, **self-supervised learning** and **detection** using different MRI modalities, but these are more generically applicable to other problems - I try to follow a modular design and development, such that networks can be deployed to different problems as necessary. I also do some work with self supervised learning methods, and have recently started to implement some building blocks for continuous learning. I prefer to organize data using `json` files so I have developed a number of scripts that allow me to achieve this (i.e. `utils/get-dataset-json.py`) and generate "dataset JSON files". By a dataset JSON file I merely mean a JSON file with the following format:

```
entry_1
|-image_0: path_to_image_0
|-image_1: path_to_image_1
|-feature_0: value_for_feature_0
|-class: class_for_entry_1
```

Then, using some minor JSON manipulation and [`MONAI`](https://monai.io/) I am able to easily construct data ingestion pipelines for training.

## Implemented methods for segmentation

* [**U-Net**](https://www.nature.com/articles/s41592-018-0261-2) - different versions are required for 2D and 3D, but here I developed a class that is able to coordinate the operations to setup both (this idea was based on the MONAI implementation of the U-Net)
* [**U-Net++**](https://pubmed.ncbi.nlm.nih.gov/32613207/) - very similar to U-Net but features [DenseNet](https://arxiv.org/abs/1608.06993)-like skip connections and skip connections between different resolutions. Also features deep supervision at the level of intermediate skip connections
* [**Anysotropic Hybrid network (AHNet)**](https://arxiv.org/abs/1711.08580) - this network is first trained to segment 2D images and some of the (enconding) layers are then transferred to 3D (mostly by either concatenating weights or adding an extra dimension to the layer).
* **Branched input U-Net (BrUNet)** - a U-Net model that has different encoders for each input channel
* [**UNETR**](https://arxiv.org/abs/2103.10504) - transformer-based U-Net
* [**SWINUNet**](https://arxiv.org/pdf/2103.14030.pdf) - transformer-based U-Net with shifted windows. Has better performance than UNETR while keeping a relatively similar complexity in terms of flops

## Implemented methods for detection

* YOLO-based network for 3d detection
* Also implemented a coarse segmentation algorithm, similar to YOLO but outputs only the object probability mask

## Implemented methods for object classification

* ResNet-based methods mostly (in `lib/modules/res_net.py`), but others can be easily constructed with the building blocks in `lib/modules/layers.py`
* [**ConvNeXt**](https://arxiv.org/abs/2201.03545) - an upgrade to CNNs that makes them comparable to vision tranformers including SWin (in `lib/modules/conv_next.py`)
* [**Vision transformer**](https://arxiv.org/abs/2010.11929v2) - A transformer, but for images
* **Factorized vision transformer** - A transformer that first processes information *within* slices (3rd spatial dimension) and only then *between* slices.

## Implemented methods for self-supervised learning

* [**BYOL**](https://arxiv.org/abs/2006.07733) - the paper that proposed a student/teacher type of setup where the teacher is nothing more than a exponential moving average of the whole model
* [**SimSiam**](https://arxiv.org/abs/2011.10566) - the paper that figured out that all you *really* need for self-supervised learning is a stop gradient on one of the encoders
* [**VICReg**](https://arxiv.org/abs/2105.04906) - the paper that figured out that all you *reaaaaally* need for self-supervised learning is a loss function capable of minimising the absence of variance and the covariance of representations and the invariance of pairs of representations for different views on the same image. This framework enables something even better - the networks for the two (or more) views can be wildly different with this loss, so there are **no** constraints on the inputs, i.e. the two "views" can come from distinctly different images paired through some other criteria (in clinical settings this can mean same individual or same disease, for instance)
* [**VICRegL**](https://arxiv.org/abs/2210.01571) - VICReg but works better for segmentation problems. Adds a term which minimises the same as VICReg 

## Code map

### Modules and networks

I have placed most of the scripts and implementations under `lib/modules/segmentation`, `lib/modules/classification`, `lib/modules/object_detection` and `lib/modules/self_supervised`. `lib/modules/layers` contains building blocks for 3D and 2D neural networks.

#### Adaptations to PyTorch Lightning

I use PyTorch Lightning to train my models as it offers a very comprehensive set of tools for optimisation. I.e. in `lib/modules/segmentation/pl.py` I have implemented some classes which inherit from the networks implemented in `lib/modules/segmentation` so that they can be trained using PyTorch Lightning. The same has been done for the self-supervised learning models (in `lib/modules/self_supervised/pl.py`) and for the classification models (in `lib/modules/classification/pl.py`).

### Segmentation loss functions

In `lib/modules/losses.py` I have coded most losses necessary based loosely on a paper by [Yeung *et al.*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8785124/) introducing the unified focal loss.

### Self-supervised loss functions

In `lib/modules/self_supervised/self_supervised.py` you can find all the relevant loss functions.

### Tests

I have included a few unit tests in `testing`. In them, I confirm that networks and modules are outputing the correct shapes and that they are compiling correctly. They are prepared to run with `pytest`, i.e. `pytest` runs all of the relevant tests.
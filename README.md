# ADeLL-MRI - a Deep-Learning Library for MRI

Here we implement and develop methods for **classification**, **segmentation**, **self-supervised learning** and **detection** using different MRI modalities, but these are more generically applicable to other problems - we try to follow a modular design and development, such that networks can be deployed to different problems as necessary. we also do some work with self supervised learning methods, and have recently started to implement some building blocks for continuous learning. we prefer to organize data using `json` files so we have developed a number of scripts that allow us to achieve this (i.e. `python -m adell_mri utils generate_dataset_json`) and generate "dataset JSON files". By a dataset JSON file we merely mean a JSON file with the following format:

```
entry_1
|-image_0: path_to_image_0
|-image_1: path_to_image_1
|-feature_0: value_for_feature_0
|-class: class_for_entry_1
```

Then, using some minor JSON manipulation and [`MONAI`](https://monai.io/) we are able to easily construct data ingestion pipelines for training.

## Installation

Installing `adell-mri` as a package can be done easily through `uv`. This can be performed inside of a `conda` (or `micromamba`) environment, but that is not necessary:

```
# creates and activates environment; these are optional! uv install a virtual env
micromamba create -n adell_env python=3.11
micromamba activate adell_env

# installs everything you need (apart from uv, which you should have installed by now)!
uv sync
```

Using these you can run `adell` from your command line as an [entrypoint](#entrypoints). Alternatively, you can still use `uv` to install everything you need to run `adell` from the command line using `uv pip install -e .` on the root folder.

### Short note on using `uv sync`

The main change you will notice if you do everything through `uv sync` is that you need to prepend any command with `uv run` as this will tell `uv` to run the command in the environment it is managing. Apart from that you are good to go!

## Implemented methods 

### Segmentation

* [**U-Net**](https://www.nature.com/articles/s41592-018-0261-2) - different versions are required for 2D and 3D, but here we developed a class that is able to coordinate the operations to setup both (this idea was based on the MONAI implementation of the U-Net)
* [**U-Net++**](https://pubmed.ncbi.nlm.nih.gov/32613207/) - very similar to U-Net but features [DenseNet](https://arxiv.org/abs/1608.06993)-like skip connections and skip connections between different resolutions. Also features deep supervision at the level of intermediate skip connections
* [**Anysotropic Hybrid network (AHNet)**](https://arxiv.org/abs/1711.08580) - this network is first trained to segment 2D images and some of the (enconding) layers are then transferred to 3D (mostly by either concatenating weights or adding an extra dimension to the layer).
* **Branched input U-Net (BrUNet)** - a U-Net model that has different encoders for each input channel
* [**UNETR**](https://arxiv.org/abs/2103.10504) - transformer-based U-Net
* [**SWINUNet**](https://arxiv.org/pdf/2103.14030.pdf) - transformer-based U-Net with shifted windows. Has better performance than UNETR while keeping a relatively similar complexity in terms of flops (**this is not currently functional and we are in the processing of figuring out why**)

### Implemented methods for detection

* YOLO-based network for 3d detection
* Also implemented a coarse segmentation algorithm, similar to YOLO but outputs only the object probability mask

### Implemented methods for classification

* Regular, VGG-like networks (just simple concatenations of convolution + activation + normalization)
* ResNet-based methods
* [**ConvNeXt**](https://arxiv.org/abs/2201.03545) - an upgrade to CNNs that makes them comparable to vision tranformers including SWin
* [**Vision transformer**](https://arxiv.org/abs/2010.11929v2) - A transformer, but for images
* **Factorized vision transformer** - A transformer that first processes information *within* slices (3rd spatial dimension) and only then *between* slices.

### Implemented methods for self-supervised learning

* [**BYOL**](https://arxiv.org/abs/2006.07733) - the paper that proposed a student/teacher type of setup where the teacher is nothing more than a exponential moving average of the whole model
* [**SimSiam**](https://arxiv.org/abs/2011.10566) - the paper that figured out that all you *really* need for self-supervised learning is a stop gradient on one of the encoders
* [**VICReg**](https://arxiv.org/abs/2105.04906) - the paper that figured out that all you *reaaaaally* need for self-supervised learning is a loss function capable of minimising the absence of variance and the covariance of representations and the invariance of pairs of representations for different views on the same image. This framework enables something even better - the networks for the two (or more) views can be wildly different with this loss, so there are **no** constraints on the inputs, i.e. the two "views" can come from distinctly different images paired through some other criteria (in clinical settings this can mean same individual or same disease, for instance)
* [**VICRegL**](https://arxiv.org/abs/2210.01571) - VICReg but works better for segmentation problems. Adds a term which minimises the same as VICReg 
* [**I-JEPA**](https://arxiv.org/abs/2301.08243) - similar to a masked auto-encoder but using a transformer architecture and masking only at the deep token features level

## Simplified library map

### Modules and networks

`adell_mri/modules/layers` contains building blocks for 3D and 2D neural networks. The remaining `adell_mri/modules/...` folders contain implementations for different applications.

#### Adaptations to PyTorch Lightning

I use PyTorch Lightning to train my models as it offers a very comprehensive set of tools for optimisation. I.e. in `adell_mri/modules/segmentation/pl.py` we have implemented some classes which inherit from the networks implemented in `adell_mri/modules/segmentation` so that they can be trained using PyTorch Lightning. The same has been done for other tasks (classification, detection, segmentation...)

### Entrypoints

A generic entrypoint has been created, accessible through `python -m adell_mri` (or `adell` if the package is installed as described in [Installation](#installation)). All entrypoints live under `adell_mri/entrypoints` and follow the same two-level dispatch pattern:

```
adell_mri <mode> <sub-command> [args...]
```

Running `adell_mri` with no arguments prints the top-level supported modes. Each mode then has its own set of sub-commands. Arguments can also be provided via a YAML parameter file using `--params_from <path.yaml>` for some endpoints.

---

#### `classification`

```
adell_mri classification {train,test,predict,explain,model_to_torchscript}
```

Standard image classification pipeline. Supports categorical (`cat`), ordinal (`ord`), U-Net encoder (`unet`), Vision Transformer (`vit`), Factorized Vision Transformer (`factorized_vit`) and VGG-like (`vgg`) network types.

- **`train`** – Cross-validated training with stratified k-folds. Supports data augmentation, label smoothing, mixup, class-weighted sampling, gradient clipping, learning-rate warm-up / decay, stochastic weight averaging (SWA), early stopping, and optional partial parameter freezing (useful for fine-tuning from a checkpoint). Logs metrics to W&B or a local CSV.
- **`test`** – Evaluates one or more checkpoints on a labelled dataset and writes bootstrap-aggregated metrics to a file.
- **`predict`** – Runs inference on unlabelled data and writes per-sample predictions (probabilities, logits, or pre-bias ordinal values) to a JSON file. Supports ensemble prediction from multiple checkpoints.
- **`explain`** – Generates saliency/attribution maps for trained models using [Captum](https://captum.ai/) (`IntegratedGradients` or `LayerGradCam`). Saves output as NIfTI/SimpleITK images alongside the originals. `explain` is implemented exclusively for the `classification` modules.
- **`model_to_torchscript`** – Traces a trained classification model to a TorchScript module for deployment.

---

#### `classification_deconfounder`

```
adell_mri classification_deconfounder {train,test,predict}
```

Variant of the classification pipeline that explicitly models and removes the effect of confounding variables (e.g. scanner, site, patient demographics) via a deconfounder network head. Accepts `--cat_confounder_keys` / `--cont_confounder_keys` for categorical and continuous confounders and `--n_features_deconfounder` to size the deconfounder. Otherwise shares the same `train` / `test` / `predict` interface as the standard classification entrypoint.

---

#### `classification_mil`

```
adell_mri classification_mil {train,test,predict}
```

Multiple-Instance Learning (MIL) classification. Treats a 3D volume as a bag of 2D slices. Supports attention-based (`MultipleInstanceClassifierPL`) and transformer-based (`TransformableTransformerPL`) aggregation methods, selectable via `--mil_method`. A pre-trained 2D feature extractor can be loaded via `--module_path`.

---

#### `classification_ensemble`

```
adell_mri classification_ensemble {train,test,predict}
```

Trains an ensemble of classifiers jointly using `GenericEnsemblePL`. Each member can be a different network type or operate on a different image key, and their outputs are combined before computing the loss. Useful for multi-modal fusion.

---

#### `segmentation`

```
adell_mri segmentation {train,test,predict,test_from_predictions}
```

3D (and 2D-in-3D) segmentation pipeline.

- **`train`** – Supports full/semi-supervised training (a combined loader can mix labelled and unlabelled data), k-fold cross-validation, multiple loss functions, partially random sampling, and optional SSL pre-training backbone initialisation.
- **`test`** – Evaluates checkpoints with sliding-window inference and computes segmentation metrics (IoU, Dice, lesion-level detection metrics via `get_lesions`).
- **`predict`** – Runs sliding-window segmentation inference and writes prediction masks to disk via `SitkWriter`.
- **`test_from_predictions`** – Re-computes metrics from predicted files.

---

#### `segmentation_from_2d_module`

```
adell_mri segmentation_from_2d_module {train}
```

Trains a 3D segmentation network that borrows its encoder from a 2D module (AHNet-style transfer). Uses `MIMUNetPL` internally.

---

#### `detection`

*This is **experimental***

```
adell_mri detection {train,predict}
```

3D object detection using a YOLO-inspired network (`YOLONet3d`). Requires a pre-computed anchor CSV (`--anchor_csv`) generated by `adell_mri utils bb_to_anchors`. Accepts bounding-box annotations via `--box_key` and `--box_class_key` in the dataset JSON.

---

#### `ssl`

```
adell_mri ssl {train_2d,train_3d,model_to_torchscript,predict_folder}
```

Self-supervised pre-training.

- **`train_2d`** – Trains a 2D SSL model (BYOL, SimSiam, VICReg, VICRegL, I-JEPA) from DICOM slices sampled on-the-fly with `SliceSampler`. Suitable for large unlabelled DICOM archives.
- **`train_3d`** – Trains a 3D SSL model from volumetric NIfTI data.
- **`model_to_torchscript`** – Exports the encoder backbone of a trained SSL model to TorchScript for downstream use.
- **`predict_folder`** – Runs a TorchScript SSL encoder over a folder of DICOM files and saves per-file feature vectors to a JSON file.

---

#### `generative`

```
adell_mri generative {train,generate}
```

Conditional / unconditional diffusion-based image generation.

- **`train`** – Trains a generative diffusion model. Supports categorical (`--cat_condition_keys`) and numerical (`--num_condition_keys`) conditioning with classifier-free guidance (controlled via `--uncondition_proba`). Uses an EMA callback and logs sample images during training.
- **`generate`** – Runs reverse diffusion from a checkpoint to produce synthetic MRI volumes. Conditioning specifications and transform parameters are automatically recovered from the checkpoint metadata.

---

#### Utilities (`utils`)

```
adell_mri utils <utility>
```

A collection of standalone data preparation and analysis scripts. Available utilities:

**Preprocessing**

| Utility | Description |
|---|---|
| `bias_field_correction` | Correct the bias field in MRI scans. |
| `merge_masks` | Merge two masks with an OR operator. |
| `resample_image` | Resample an image to a target spacing. |
| `resample_volumes_and_masks` | Resample volumes and their masks to a target spacing. |

**Dataset management**

| Utility | Description |
|---|---|
| `generate_dataset_json` | Create a dataset JSON file with image paths and bounding boxes. |
| `generate_dicom_dataset_json` | Create a dataset JSON file from a DICOM archive. |
| `generate_image_dataset_json` | Create a dataset JSON file from generic image files. |
| `generate_json_from_csv` | Convert a CSV into a hierarchical dataset JSON. |
| `merge_json_datasets` | Merge two JSON datasets (conflict resolution via suffixes). |
| `fill_with_condition` | Create empty masks for a given image key when a condition is met. |
| `get_test_set_and_folds` | Split dataset entries into a test set and cross-validation folds. |
| `get_temporal_test_set_and_folds` | Split by a date key into prospective test set and folds. |
| `get_image_examples` | Produce image examples from a DICOM dataset after transforms. |
| `get_mask_coordinates` | Produce a JSON of mask coordinates after spatial transforms. |
| `remove_constant_masks` | Remove empty/constant masks from a dataset JSON. |
| `bb_to_anchors` | Compute detection anchors from bounding-box annotations. |
| `bb_to_distances` | Compute minimum distances between bounding boxes. |
| `describe_sitk` | Print SimpleITK image properties. |
| `describe_dicom_dataset` | Print general statistics for a DICOM dataset. |
| `inspect_dicom_dataset` | List entries with NaN/infinite values in a DICOM dataset. |

**Statistics**

| Utility | Description |
|---|---|
| `compare_masks` | Compute IoU between masks in two folders with matching identifiers. |
| `get_label_size` | Print the size of labels in a folder of segmentation masks. |
| `match_to_mask` | Determine which MRI sequence was most likely used as the mask template. |

**Other**

| Utility | Description |
|---|---|
| `random_image_panel` | Generate a panel of random images from a DICOM folder. |
| `test_traced_model` | Test a JIT-traced model with an input of a given shape. |

This creates a consistent way of entering different scripts. All entrypoints are specified in `adell_mri/entrypoints`.

### Tests

I have included a few unit tests in `testing`. In them, we confirm that networks and modules are outputing the correct shapes and that they are compiling correctly. They are prepared to run with `pytest`, i.e. `pytest` runs all of the relevant tests.

## CCIG publications using `adell`

* Rodrigues NM, Almeida JG, Verde ASC, Gaivão AM, Bilreiro C, Santiago I, Ip J, Belião S, Moreno R, Matos C, Vanneschi L, Tsiknakis M, Marias K, Regge D, Silva S; ProCAncer-I Consortium; Papanikolaou N. [Analysis of domain shift in whole prostate gland, zonal and lesions segmentation and detection, using multicentric retrospective data.](https://pubmed.ncbi.nlm.nih.gov/38442555/) Comput Biol Med. 2024 Mar 2;171:108216. doi: 10.1016/j.compbiomed.2024.108216. Epub ahead of print. PMID: 38442555.
* de Almeida JG, Rodrigues NM, Castro Verde AS, Mascarenhas Gaivão A, Bilreiro C, Santiago I, Ip J, Belião S, Matos C, Silva S, Tsiknakis M, Marias K, Regge D, Papanikolaou N; ProCAncer-I Consortium. [Impact of Scanner Manufacturer, Endorectal Coil Use, and Clinical Variables on Deep Learning-assisted Prostate Cancer Classification Using Multiparametric MRI](https://pubmed.ncbi.nlm.nih.gov/39841063/). Radiol Artif Intell. 2025 May;7(3):e230555. doi: 10.1148/ryai.230555. PMID: 39841063.
* de Almeida JG, Castro Verde AS, Mascarenhas Gaivão A, Bilreiro C, Santiago I, Ip J, Belião S, Matos C, Tsiknakis M, Marias K, Regge D; ProCAncer-I Consortium; Papanikolaou N. [Self-supervised learning leads to improved performance in biparametric prostate MRI classification](https://pubmed.ncbi.nlm.nih.gov/41192119/). Comput Biol Med. 2025 Nov;198(Pt B):111262. doi: 10.1016/j.compbiomed.2025.111262. Epub 2025 Nov 4. PMID: 41192119.
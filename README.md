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

A generic entrypoint has been created, this can be accessed through `python -m adell_mri` (or `adell` if you have installed this package as described in the [installation](#installation)). Running this produces:

```
        Supported modes: ['classification', 'classification_deconfounder', 'classification_mil', 'classification_ensemble', 'generative', 'segmentation', 'segmentation_from_2d_module', 'ssl', 'detection', 'utils']
```

And specifying different modes leads to (for classification, for example - `python -m adell_mri classification`):

```
        Supported modes: ['train', 'test', 'predict']
```

Finally, upon further specification (`python -m adell_mri classification train`):

```
usage: __main__.py [-h] [--net_type {cat,ord,unet,vit,factorized_vit,vgg}] [--params_from PARAMS_FROM] --dataset_json DATASET_JSON
                   --image_keys IMAGE_KEYS [IMAGE_KEYS ...] [--clinical_feature_keys CLINICAL_FEATURE_KEYS [CLINICAL_FEATURE_KEYS ...]]
                   [--label_keys LABEL_KEYS] [--mask_key MASK_KEY] [--image_masking] [--image_crop_from_mask]
                   [--t2_keys T2_KEYS [T2_KEYS ...]] [--adc_keys ADC_KEYS [ADC_KEYS ...]] --possible_labels POSSIBLE_LABELS
                   [POSSIBLE_LABELS ...] [--positive_labels POSITIVE_LABELS [POSITIVE_LABELS ...]]
                   [--label_groups LABEL_GROUPS [LABEL_GROUPS ...]] [--cache_rate CACHE_RATE]
                   [--target_spacing TARGET_SPACING [TARGET_SPACING ...]] [--pad_size PAD_SIZE [PAD_SIZE ...]]
                   [--crop_size CROP_SIZE [CROP_SIZE ...]] [--subsample_size SUBSAMPLE_SIZE]
                   [--subsample_training_data SUBSAMPLE_TRAINING_DATA] [--filter_on_keys FILTER_ON_KEYS [FILTER_ON_KEYS ...]]
                   [--val_from_train VAL_FROM_TRAIN] --config_file CONFIG_FILE [--dev DEV] [--n_workers N_WORKERS] [--seed SEED]
                   [--augment AUGMENT [AUGMENT ...]] [--label_smoothing LABEL_SMOOTHING] [--mixup_alpha MIXUP_ALPHA]
                   [--partial_mixup PARTIAL_MIXUP] [--max_epochs MAX_EPOCHS] [--n_folds N_FOLDS] [--folds FOLDS [FOLDS ...]]
                   [--excluded_ids EXCLUDED_IDS [EXCLUDED_IDS ...]]
                   [--excluded_ids_from_training_data EXCLUDED_IDS_FROM_TRAINING_DATA [EXCLUDED_IDS_FROM_TRAINING_DATA ...]]
                   [--checkpoint_dir CHECKPOINT_DIR] [--checkpoint_name CHECKPOINT_NAME] [--checkpoint CHECKPOINT [CHECKPOINT ...]]
                   [--delete_checkpoints] [--freeze_regex FREEZE_REGEX [FREEZE_REGEX ...]]
                   [--not_freeze_regex NOT_FREEZE_REGEX [NOT_FREEZE_REGEX ...]]
                   [--exclude_from_state_dict EXCLUDE_FROM_STATE_DICT [EXCLUDE_FROM_STATE_DICT ...]] [--resume_from_last] [--monitor MONITOR]
                   [--project_name PROJECT_NAME] [--summary_dir SUMMARY_DIR] [--summary_name SUMMARY_NAME] [--metric_path METRIC_PATH]
                   [--resume {allow,must,never,auto,none}] [--warmup_steps WARMUP_STEPS] [--start_decay START_DECAY]
                   [--dropout_param DROPOUT_PARAM] [--accumulate_grad_batches ACCUMULATE_GRAD_BATCHES] [--gradient_clip_val GRADIENT_CLIP_VAL]
                   [--early_stopping EARLY_STOPPING] [--swa] [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
                   [--class_weights CLASS_WEIGHTS [CLASS_WEIGHTS ...]] [--weighted_sampling] [--correct_classification_bias]

options:
  -h, --help            show this help message and exit
  --net_type {cat,ord,unet,vit,factorized_vit,vgg}
                        Classification type
  --params_from PARAMS_FROM
                        Parameter path used to retrieve values for the CLI (can be a path to a YAML file or 'dvc' to retrieve dvc params)
  --dataset_json DATASET_JSON
                        JSON containing dataset information
  --image_keys IMAGE_KEYS [IMAGE_KEYS ...]
                        Image keys in the dataset JSON.
  --clinical_feature_keys CLINICAL_FEATURE_KEYS [CLINICAL_FEATURE_KEYS ...]
                        Tabular clinical feature keys in the dataset JSON.
  --label_keys LABEL_KEYS
                        Label keys in the dataset JSON.
  --mask_key MASK_KEY   Mask key in dataset JSON
  --image_masking       Uses mask_key to mask the rest of the image.
  --image_crop_from_mask
                        Crops image using mask_key.
  --t2_keys T2_KEYS [T2_KEYS ...]
                        Image keys corresponding to T2.
  --adc_keys ADC_KEYS [ADC_KEYS ...]
                        Image keys corresponding to ADC.
  --possible_labels POSSIBLE_LABELS [POSSIBLE_LABELS ...]
                        All the possible labels in the data.
  --positive_labels POSITIVE_LABELS [POSITIVE_LABELS ...]
                        Labels that should be considered positive (binarizes labels)
  --label_groups LABEL_GROUPS [LABEL_GROUPS ...]
                        Label groups for classification.
  --cache_rate CACHE_RATE
                        Rate of samples to be cached
  --target_spacing TARGET_SPACING [TARGET_SPACING ...]
                        Resamples all images to target spacing
  --pad_size PAD_SIZE [PAD_SIZE ...]
                        Size of central padded image after resizing (if none is specified then no padding is performed).
  --crop_size CROP_SIZE [CROP_SIZE ...]
                        Size of central crop after resizing (if none is specified then no cropping is performed).
  --subsample_size SUBSAMPLE_SIZE
                        Subsamples data to a given size
  --subsample_training_data SUBSAMPLE_TRAINING_DATA
                        Subsamples training data by this fraction (for learning curves)
  --filter_on_keys FILTER_ON_KEYS [FILTER_ON_KEYS ...]
                        Filters the dataset based on a set of specific key:value pairs.
  --val_from_train VAL_FROM_TRAIN
                        Uses this fraction of training data as a validation set during training
  --config_file CONFIG_FILE
                        Path to network configuration file (yaml)
  --dev DEV             Device for PyTorch training
  --n_workers N_WORKERS
                        No. of workers
  --seed SEED           Random seed
  --augment AUGMENT [AUGMENT ...]
                        Use data augmentations
  --label_smoothing LABEL_SMOOTHING
                        Label smoothing value
  --mixup_alpha MIXUP_ALPHA
                        Alpha for mixup
  --partial_mixup PARTIAL_MIXUP
                        Applies mixup only to this fraction of the batch
  --max_epochs MAX_EPOCHS
                        Maximum number of training epochs
  --n_folds N_FOLDS     Number of validation folds
  --folds FOLDS [FOLDS ...]
                        Comma-separated IDs to be used in each space-separated fold
  --excluded_ids EXCLUDED_IDS [EXCLUDED_IDS ...]
                        Comma separated list of IDs to exclude.
  --excluded_ids_from_training_data EXCLUDED_IDS_FROM_TRAINING_DATA [EXCLUDED_IDS_FROM_TRAINING_DATA ...]
                        Comma separated list of IDs to exclude from training data.
  --checkpoint_dir CHECKPOINT_DIR
                        Path to directory where checkpoints will be saved.
  --checkpoint_name CHECKPOINT_NAME
                        Checkpoint ID.
  --checkpoint CHECKPOINT [CHECKPOINT ...]
                        Resumes training from or tests/predicts with these checkpoint.
  --delete_checkpoints  Deletes checkpoints after training (keeps only metrics).
  --freeze_regex FREEZE_REGEX [FREEZE_REGEX ...]
                        Matches parameter names and freezes them.
  --not_freeze_regex NOT_FREEZE_REGEX [NOT_FREEZE_REGEX ...]
                        Matches parameter names and skips freezing them (overrides --freeze_regex)
  --exclude_from_state_dict EXCLUDE_FROM_STATE_DICT [EXCLUDE_FROM_STATE_DICT ...]
                        Regex to exclude parameters from state dict in --checkpoint
  --resume_from_last    Resumes from the last checkpoint stored for a given fold.
  --monitor MONITOR     Metric that is monitored to determine the best checkpoint.
  --project_name PROJECT_NAME
                        Wandb project name.
  --summary_dir SUMMARY_DIR
                        Path to summary directory (for wandb).
  --summary_name SUMMARY_NAME
                        Summary name.
  --metric_path METRIC_PATH
                        Path to file with CV metrics + information.
  --resume {allow,must,never,auto,none}
                        Whether wandb project should be resumed (check https://docs.wandb.ai/ref/python/init for more details).
  --warmup_steps WARMUP_STEPS
                        Number of warmup steps/epochs (if SWA is triggered it starts after this number of steps).
  --start_decay START_DECAY
                        Step at which decay starts. Defaults to starting right after warmup ends.
  --dropout_param DROPOUT_PARAM
                        Parameter for dropout.
  --accumulate_grad_batches ACCUMULATE_GRAD_BATCHES
                        Number batches to accumulate before backpropgating gradient
  --gradient_clip_val GRADIENT_CLIP_VAL
                        Value for gradient clipping
  --early_stopping EARLY_STOPPING
                        No. of checks before early stop (defaults to no early stop).
  --swa                 Use stochastic gradient averaging.
  --learning_rate LEARNING_RATE
                        Overrides learning rate in config file
  --batch_size BATCH_SIZE
                        Batch size
  --class_weights CLASS_WEIGHTS [CLASS_WEIGHTS ...]
                        Class weights (by alphanumeric order).
  --weighted_sampling   Samples according to class proportions.
  --correct_classification_bias
                        Sets the final classification bias to log(pos/neg).
```

This creates a consistent way of entering different scripts. All entrypoints are specified in `adell_mri/entrypoints`.

### Tests

I have included a few unit tests in `testing`. In them, we confirm that networks and modules are outputing the correct shapes and that they are compiling correctly. They are prepared to run with `pytest`, i.e. `pytest` runs all of the relevant tests.

## To-do

* <del>Change dataset generation to entrypoint</del>
* <del>Create minimal installer</del>

## CCIG publications using `adell`

* Rodrigues NM, Almeida JG, Verde ASC, Gaivão AM, Bilreiro C, Santiago I, Ip J, Belião S, Moreno R, Matos C, Vanneschi L, Tsiknakis M, Marias K, Regge D, Silva S; ProCAncer-I Consortium; Papanikolaou N. [Analysis of domain shift in whole prostate gland, zonal and lesions segmentation and detection, using multicentric retrospective data.](https://pubmed.ncbi.nlm.nih.gov/38442555/) Comput Biol Med. 2024 Mar 2;171:108216. doi: 10.1016/j.compbiomed.2024.108216. Epub ahead of print. PMID: 38442555.
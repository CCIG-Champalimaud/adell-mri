from argparse import ArgumentParser
from typing import List
from copy import deepcopy

argument_factory = {
    # dataset and data
    "dataset_json": dict(
        type=str, help="JSON containing dataset information", required=True
    ),
    "sequence_paths": dict(
        type=str, nargs="+", help="Path to sequence", default=None
    ),
    "image_keys": dict(
        type=str,
        nargs="+",
        required=True,
        help="Image keys in the dataset JSON.",
    ),
    "label_keys": dict(
        type=str,
        default="image_labels",
        help="Label keys in the dataset JSON.",
    ),
    "clinical_feature_keys": dict(
        type=str,
        nargs="+",
        default=None,
        help="Tabular clinical feature keys in the dataset JSON.",
    ),
    "t2_keys": dict(
        type=str,
        nargs="+",
        default=None,
        help="Image keys corresponding to T2.",
    ),
    "adc_keys": dict(
        type=str,
        nargs="+",
        default=None,
        help="Image keys corresponding to ADC.",
    ),
    "cat_condition_keys": dict(
        default=None,
        nargs="+",
        help="Label keys in the dataset JSON for categorical variables \
            (applied with classifier-free guidance).",
    ),
    "num_condition_keys": dict(
        default=None,
        nargs="+",
        help="Label keys in the dataset JSON for numerical variables \
            (applied with classifier-free guidance).",
    ),
    "filter_on_keys": dict(
        default=[],
        nargs="+",
        help="Filters the dataset based on a set of specific key:value pairs.",
    ),
    "filter_is_optional": dict(
        default=False,
        action="store_true",
        help="Considers the fields in the filter to be optional",
    ),
    "possible_labels": dict(
        type=str,
        nargs="+",
        required=True,
        help="All the possible labels in the data.",
    ),
    "positive_labels": dict(
        type=str,
        nargs="+",
        default=None,
        help="Labels that should be considered positive (binarizes labels)",
    ),
    "label_groups": dict(
        type=str,
        nargs="+",
        default=None,
        help="Label groups for classification.",
    ),
    # data sampling and splitting
    "cache_rate": dict(
        type=float, help="Rate of samples to be cached", default=1.0
    ),
    "subsample_size": dict(
        type=int, help="Subsamples data to a given size", default=None
    ),
    "subsample_training_data": dict(
        type=float,
        help="Subsamples training data by this fraction (for learning curves)",
        default=None,
    ),
    "val_from_train": dict(
        default=None,
        type=float,
        help="Uses this fraction of training data as a validation set \
            during training",
    ),
    "use_val_as_train_val": dict(
        action="store_true",
        default=False,
        help="Use validation set as training validation set.",
    ),
    "excluded_ids": dict(
        type=str,
        default=None,
        nargs="+",
        help="Comma separated list of IDs to exclude.",
    ),
    "excluded_ids_from_training_data": dict(
        type=str,
        default=None,
        nargs="+",
        help="Comma separated list of IDs to exclude from training data.",
    ),
    "n_folds": dict(default=5, type=int, help="Number of validation folds"),
    "folds": dict(
        type=str,
        default=None,
        nargs="+",
        help="Comma-separated IDs to be used in each space-separated fold",
    ),
    # data processing
    "target_spacing": dict(
        default=None,
        nargs="+",
        type=float,
        help="Resamples all images to target spacing",
    ),
    "pad_size": dict(
        default=None,
        type=float,
        nargs="+",
        help="Size of central padded image after resizing (if none is specified\
            then no padding is performed).",
    ),
    "crop_size": dict(
        default=None,
        type=float,
        nargs="+",
        help="Size of central crop after resizing (if none is specified then\
            no cropping is performed).",
    ),
    "resize_size": dict(
        type=float, nargs="+", default=None, help="Input size for network"
    ),
    "resize_keys": dict(
        type=str,
        nargs="+",
        default=None,
        help="Keys that will be resized to input size",
    ),
    # network specification
    "config_file": dict(
        type=str,
        help="Path to network configuration file (yaml)",
        required=True,
    ),
    "config_files": dict(
        type=str,
        required=False,
        help="Paths to network configuration file (yaml; size 1 or same size as\
            net types)",
    ),
    "ensemble_config_file": dict(required=True, help="Configures ensemble."),
    "overrides": dict(
        help="Overrides parameters in config yaml (hydra)",
        default=None,
        nargs="+",
    ),
    "warmup_steps": dict(
        type=float,
        default=0.0,
        help="Number of warmup steps/epochs (if SWA is triggered it starts \
            after this number of steps).",
    ),
    "start_decay": dict(
        type=float,
        default=None,
        help="Step at which decay starts. Defaults to starting right after \
            warmup ends.",
    ),
    "n_classes": dict(type=int, default=2, help="Number of classes"),
    # collects params from yaml files
    "params_from": dict(
        type=str,
        default=None,
        help="Parameter path used to retrieve values for the CLI (can be a path\
            to a YAML file or 'dvc' to retrieve dvc params)",
    ),
    # device and machine
    "dev": dict(default="cpu", type=str, help="Device for PyTorch training"),
    "n_workers": dict(default=0, type=int, help="No. of workers"),
    "seed": dict(default=42, type=int, help="Random seed"),
    # checkpointing/checkpoint loading
    "checkpoint_dir": dict(
        type=str,
        default=None,
        help="Path to directory where checkpoints will be saved.",
    ),
    "checkpoint_name": dict(type=str, default=None, help="Checkpoint ID."),
    "checkpoint": dict(
        type=str,
        default=None,
        nargs="+",
        help="Resumes training from or tests/predicts with these checkpoint.",
    ),
    "checkpoint_ensemble": dict(
        type=str,
        default=None,
        nargs="+",
        help="Resumes training from these checkpoint. For each model\
            in the ensemble, a checkpoint should be provided in a comma \
            separated list and folds separated by space. I.e., for 2 folds and\
            3 models in the ensemble: \
            model0_f0,model1_f0,model2_f0 model0_f1,model1_f1,model2_f1",
    ),
    "freeze_regex": dict(
        type=str,
        default=None,
        nargs="+",
        help="Matches parameter names and freezes them.",
    ),
    "not_freeze_regex": dict(
        type=str,
        default=None,
        nargs="+",
        help="Matches parameter names and skips freezing them (overrides \
            --freeze_regex)",
    ),
    "exclude_from_state_dict": dict(
        type=str,
        default=None,
        nargs="+",
        help="Regex to exclude parameters from state dict in --checkpoint",
    ),
    "resume_from_last": dict(
        action="store_true",
        help="Resumes from the last checkpoint stored for a given fold.",
    ),
    "delete_checkpoints": dict(
        action="store_true",
        help="Deletes checkpoints after training (keeps only metrics).",
    ),
    # logging/summary
    "logger_type": dict(
        type=str,
        default="wandb",
        choices=["wandb", "mlflow"],
        help="Type of logger that will be used. mlflow requires defining \
            --tracking_uri.",
    ),
    "project_name": dict(
        type=str,
        default=None,
        help="Wandb project name or MLFlow experiment ID.",
    ),
    "monitor": dict(
        type=str,
        default="val_loss",
        help="Metric that is monitored to determine the best checkpoint.",
    ),
    "summary_dir": dict(
        type=str,
        default="summaries",
        help="Path to summary directory (for wandb).",
    ),
    "summary_name": dict(
        type=str,
        default=None,
        help="Summary name for logging.",
    ),
    "tracking_uri": dict(
        type=str,
        default=None,
        help="Tracking URI for MLFlow logging.",
    ),
    "metric_path": dict(
        type=str,
        default="metrics.csv",
        help="Path to file with CV metrics + information.",
    ),
    "resume": dict(
        type=str,
        default="allow",
        choices=["allow", "must", "never", "auto", "none"],
        help="Whether wandb project should be resumed (check \
            https://docs.wandb.ai/ref/python/init for more details).",
    ),
    "log_model": dict(
        type=str,
        default=False,
        action="store_true",
        help="Stores models with loggers as artefacts",
    ),
    "output_path": dict(
        type=str, default="output.csv", help="Path to output file."
    ),
    # training
    "max_epochs": dict(
        default=100, type=int, help="Maximum number of training epochs"
    ),
    "augment": dict(
        type=str, nargs="+", default=[], help="Use data augmentations"
    ),
    "label_smoothing": dict(
        default=None, type=float, help="Label smoothing value"
    ),
    "mixup_alpha": dict(default=None, type=float, help="Alpha for mixup"),
    "partial_mixup": dict(
        default=None,
        type=float,
        help="Applies mixup only to this fraction of the batch",
    ),
    "swa": dict(
        action="store_true",
        default=False,
        help="Use stochastic gradient averaging.",
    ),
    "early_stopping": dict(
        type=int,
        default=None,
        help="No. of checks before early stop (defaults to no early stop).",
    ),
    "cosine_decay": dict(
        action="store_true",
        default=False,
        help="Decreases the LR using cosine decay.",
    ),
    # training (network)
    "dropout_param": dict(
        type=float, default=0.1, help="Parameter for dropout."
    ),
    "batch_size": dict(type=int, default=2, help="Batch size"),
    "learning_rate": dict(
        type=float,
        default=0.0001,
        help="Overrides learning rate in config file",
    ),
    # trainer (lightning)
    "precision": dict(type=str, default="32", help="Floating point precision"),
    "check_val_every_n_epoch": dict(
        default=5, type=int, help="Validation check frequency"
    ),
    "gradient_clip_val": dict(
        default=0.0, type=float, help="Value for gradient clipping"
    ),
    "accumulate_grad_batches": dict(
        default=1,
        type=int,
        help="Number batches to accumulate before backpropgating gradient",
    ),
    # classification-specific
    "class_weights": dict(
        type=str,
        nargs="+",
        default=None,
        help="Class weights (by alphanumeric order).",
    ),
    "weighted_sampling": dict(
        action="store_true",
        default=False,
        help="Samples according to class proportions.",
    ),
    "correct_classification_bias": dict(
        action="store_true",
        default=False,
        help="Sets the final classification bias to log(pos/neg).",
    ),
    "classification_net_type": dict(
        default="cat",
        choices=["cat", "ord", "unet", "vit", "factorized_vit", "vgg"],
        help="Classification type",
    ),
    "classification_net_types": dict(
        default="cat",
        choices=["cat", "ord", "unet", "vit", "factorized_vit", "vgg"],
        help="Classification types",
    ),
    "image_masking": dict(
        action="store_true",
        default=False,
        help="Uses mask_key to mask the rest of the image.",
    ),
    "image_crop_from_mask": dict(
        action="store_true", default=False, help="Crops image using mask_key."
    ),
    # testing/prediction specific
    "one_to_one": dict(
        action="store_true",
        help="Tests the checkpoint only on the corresponding test_ids set",
    ),
    # testing specific
    "test_ids": dict(
        type=str,
        default=None,
        nargs="+",
        help="Comma-separated IDs to be used in each test",
    ),
    "test_checkpoints": dict(
        type=str, default=None, nargs="+", help="Test using these checkpoints."
    ),
    # prediction specific
    "prediction_ids": dict(
        type=str,
        default=None,
        nargs="+",
        help="Comma-separated IDs to be used in each test",
    ),
    "prediction_type": dict(
        action="store",
        default="probability",
        choices=["probability", "logit", "features"],
        help="Returns either probability the classification probability or the\
            features in the last layer.",
    ),
    "prediction_checkpoints": dict(
        type=str,
        default=None,
        nargs="+",
        help="Predict using these checkpoints.",
    ),
    # mil-specific
    "mil_method": dict(
        required=True,
        choices=["standard", "transformer"],
        help="Multiple instance learning method name.",
    ),
    "module_path": dict(required=True, help="Path to torchscript module"),
    "module_paths": dict(
        required=False,
        nargs="+",
        help="Paths to torchscript modules",
    ),
    # detection-specific
    "box_key": dict(type=str, default="boxes", help="Box key in dataset JSON"),
    "box_class_key": dict(
        type=str, default="box_classes", help="Box class key in dataset JSON"
    ),
    "shape_key": dict(
        type=str, default="shape", help="Shape key in dataset JSON"
    ),
    "mask_key": dict(type=str, default=None, help="Mask key in dataset JSON"),
    "mask_mode": dict(
        type=str,
        default="mask_is_labels",
        choices=["mask_is_labels", "infer_labels", "single_object"],
        help="If using mask_key, defines how boxes are inferred. mask_is_labels\
            uses the mask as labels; infer_labels infers connected components \
            using skimage.measure.label; single_object assumes the mask represents\
            a single, not necessarily connected, object.",
    ),
    "anchor_csv": dict(
        type=str, required=True, help="Path to CSV file containing anchors"
    ),
    "min_anchor_area": dict(
        type=float, default=None, help="Minimum anchor area (filters anchors)"
    ),
    "detection_net_type": dict(
        default="cat", choices=["yolo"], help="Network type"
    ),
    "iou_threshold": dict(
        type=float, default=0.5, help="IoU threshold for pred-gt overlaps."
    ),
    # detection/segmentation-specific
    "loss_gamma": dict(default=2.0, type=float, help="Gamma for focal loss"),
    "loss_comb": dict(
        default=0.5, type=float, help="Relative weight for combined losses"
    ),
    "loss_scale": dict(
        default=1.0, type=float, help="Loss scale (helpful for 16bit trainign"
    ),
    # segmentation-specific
    "skip_keys": dict(
        type=str,
        default=None,
        nargs="+",
        help="Key for image in the dataset JSON that is concatenated to the \
            skip connections.",
    ),
    "skip_mask_keys": dict(
        type=str,
        nargs="+",
        default=None,
        help="Key for mask in the dataset JSON that is appended to the skip \
            connections (can be useful for including prior mask as feature).",
    ),
    "mask_image_keys": dict(
        type=str,
        nargs="+",
        default=None,
        help="Keys corresponding to input images which are segmentation masks",
    ),
    "feature_keys": dict(
        type=str,
        nargs="+",
        default=None,
        help="Keys corresponding to tabular features in the JSON dataset",
    ),
    "mask_keys": dict(
        type=str, default=None, help="Mask keys in dataset JSON"
    ),
    "random_crop_size": dict(
        default=None,
        type=float,
        nargs="+",
        help="Size of crop with random centre (after padding/cropping).",
    ),
    "n_crops": dict(default=1, type=int, help="Number of random crops."),
    "bottleneck_classification": dict(
        action="store_true",
        help="Predicts the maximum class in the output using the bottleneck \
            features.",
    ),
    "deep_supervision": dict(
        action="store_true", help="Triggers deep supervision."
    ),
    "constant_ratio": dict(
        type=float,
        default=None,
        help="If there are masks with only one value, defines how many are\
            included relatively to masks with more than one value.",
    ),
    "missing_to_empty": dict(
        type=str,
        nargs="+",
        choices=["image", "mask", "from_path"],
        help="If some images or masks are missing, assume they are empty \
            tensors.",
    ),
    "segmentation_net_type": dict(
        choices=["unet", "unetpp", "brunet", "unetr", "swin"],
        default="unet",
        help="Specifies which UNet model is used",
    ),
    "res_config_file": dict(
        action="store",
        default=None,
        help="Uses a ResNet as a backbone (depths are inferred from this). \
            This config file is then used to parameterise the ResNet.",
    ),
    "encoder_checkpoint": dict(
        action="store",
        default=None,
        nargs="+",
        help="Checkpoint for encoder backbone checkpoint",
    ),
    "lr_encoder": dict(
        action="store",
        default=None,
        type=float,
        help="Sets learning rate for encoder.",
    ),
    "dataset_iterations_per_epoch": dict(
        default=1.0, type=float, help="Number of dataset iterations per epoch"
    ),
    "picai_eval": dict(
        action="store_true", help="Validates model using PI-CAI metrics."
    ),
    "sliding_window_size": dict(
        default=None, type=int, nargs="+", help="Size of sliding window."
    ),
    "keep_largest_connected_component": dict(
        action="store_true", help="Keeps only the largest connected component."
    ),
    "flip": dict(action="store_true", help="Flips before predicting."),
    "per_sample": dict(
        action="store_true",
        help="Also calculates metrics on a per sample basis.",
    ),
    "segmentation_prediction_mode": dict(
        choices=["image", "probs", "deep_features", "bounding_box"],
        default="image",
    ),
    "threshold": dict(
        default=0.5, type=float, help="Sets threshold for positive class"
    ),
    # ssl-specific
    "jpeg_dataset": dict(
        action="store_true",
        help="Whether the dataset simply contains a paragraph-separated list of\
            paths. This is helpful to avoid the JSON structure",
    ),
    "num_samples": dict(
        type=float,
        default=None,
        help="Number of samples per epoch for JPEG dataset (must be specified)\
            if the --jpeg_dataset flag is used",
    ),
    "train_pids": dict(
        default=None,
        type=str,
        nargs="+",
        help="IDs in dataset_json used for training",
    ),
    "scaled_crop_size": dict(
        default=None,
        type=int,
        nargs="+",
        help="Crops a region with at least a quarter of the specified size \
            and then resizes them image to this size.",
    ),
    "different_crop": dict(
        default=False,
        action="store_true",
        help="Uses different crops to make views.",
    ),
    "ssl_net_type": dict(
        choices=["resnet", "unet_encoder", "convnext", "vit"],
        help="Which network should be trained.",
    ),
    "ssl_method": dict(
        type=str,
        choices=["simsiam", "byol", "simclr", "vicreg", "vicregl", "ijepa"],
        help="SSL method",
    ),
    "unet_encoder": dict(action="store_true", help="Trains a UNet encoder"),
    "max_slices": dict(
        type=int, default=None, help="Excludes studies with over max_slices"
    ),
    "ema": dict(
        action="store_true",
        help="Includes exponential moving average teacher (like BYOL)",
    ),
    "stop_gradient": dict(
        action="store_true", help="Stops gradient for teacher"
    ),
    "n_series_iterations": dict(
        default=2,
        type=int,
        help="Number of iterations over each series per epoch",
    ),
    "n_transforms": dict(
        default=3, type=int, help="Number of augmentations for each image"
    ),
    "steps_per_epoch": dict(
        default=None, type=int, help="Number of steps per epoch"
    ),
    # diffusion-specific
    "diffusion_steps": dict(
        type=int, default=1000, help="Number of diffusion steps"
    ),
    "n_samples_gen": dict(
        type=int, default=1000, help="Number of samples to generate."
    ),
    "fill_missing_with_placeholder": dict(
        type=str,
        default=None,
        nargs="+",
        help="Fills missing keys with value (key1:value1, key2:value2)",
    ),
    # deconfounded classifier-specific
    "cat_confounder_keys": dict(
        type=str,
        default=None,
        nargs="+",
        help="Keys corresponding to categorical confounder.",
    ),
    "cont_confounder_keys": dict(
        type=str,
        default=None,
        nargs="+",
        help="Keys corresponding to continuous confounder.",
    ),
    # semi-supervised segmentation
    "semi_supervised": dict(
        action="store_true",
        default=False,
        help="Uses images without annotations for self-supervision.",
    ),
    # classification ensemble
    "branched": dict(
        action="store_true",
        default=False,
        help="One backbone for each input image.",
    ),
    # ensemble prediction
    "ensemble": dict(
        action="store",
        default=None,
        help="Ensembles predictions from different checkpoints.",
        choices=["mean"],
    ),
}


class Parser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_argument_by_key(self, key_list: List[str]):
        for key in key_list:
            new_kwargs = {}
            if isinstance(key, (tuple, list)):
                if len(key) > 2:
                    new_kwargs = key[2]
                real_key, key = key[0], key[1]
            else:
                real_key = key
            add_arg_kwargs = deepcopy(argument_factory[real_key])
            for k in new_kwargs:
                add_arg_kwargs[k] = new_kwargs[k]
            self.add_argument(f"--{key}", dest=key, **add_arg_kwargs)

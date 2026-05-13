import sys
from copy import deepcopy
from pathlib import Path

import monai
import numpy as np
import SimpleITK as sitk
import torch
from captum.attr import IntegratedGradients, LayerGradCam
from monai.data import MetaTensor
from monai.transforms.utils import allow_missing_keys_mode
from tqdm import tqdm

from adell_mri.entrypoints.assemble_args import Parser
from adell_mri.modules.classification.losses import OrdinalSigmoidalLoss
from adell_mri.modules.config_parsing import parse_config_cat, parse_config_unet
from adell_mri.transform_factory.transforms import ClassificationTransforms
from adell_mri.utils.dataset import Dataset
from adell_mri.utils.network_factories import get_classification_network
from adell_mri.utils.parser import get_params, merge_args, parse_ids
from adell_mri.utils.python_logging import get_logger
from adell_mri.utils.torch_utils import get_generator_and_rng

logger = get_logger(__name__)


class OrdinalRankWrapper(torch.nn.Module):
    """
    Wraps an ordinal classification model so that the output is a single
    scalar representing the predicted rank (sum of sigmoid outputs).
    This makes the output compatible with attribution methods that expect
    a scalar or per-class output without requiring a target specification.
    """

    def __init__(self, model):
        """
        Initializes the wrapper.

        Args:
            model: The ordinal classification model to wrap.
        """
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)
        probs = torch.sigmoid(logits)
        return probs.sum(dim=-1, keepdim=True)


class BinaryOutputWrapper(torch.nn.Module):
    """
    Wraps a binary classification model (single logit output) so that
    the output has two columns [1-p, p], making it compatible with captum
    attribution methods that accept a target class index.
    """

    def __init__(self, model):
        """
        Initializes the wrapper.

        Args:
            model: The binary classification model to wrap.
        """
        super().__init__()
        self.model = model

    def forward(self, x):
        logit = self.model(x)
        prob = torch.sigmoid(logit)
        if prob.dim() == 1:
            prob = prob.unsqueeze(-1)
        return torch.cat([1 - prob, prob], dim=-1)


def get_last_conv_layer(
    network: torch.nn.Module, net_type: str
) -> torch.nn.Module:
    """
    Heuristic to retrieve the last convolutional layer suitable for
    GradCAM from the different network architectures in this project.

    Args:
        network: The network model.
        net_type: The type of network.

    Returns:
        The last convolutional layer.
    """
    if net_type == "unet":
        ops = list(network.encoding_operations)
        return ops[-1][0]
    elif net_type == "vgg":
        if hasattr(network, "network"):
            return network.network.conv3
        return network.conv3
    elif net_type in ("cat", "ord"):
        inner = network.network if hasattr(network, "network") else network
        fe = inner.feature_extraction
        if hasattr(fe, "operations"):
            return fe.operations[-1]
        if hasattr(fe, "res_blocks"):
            return fe.res_blocks[-1]
        children = list(fe.children())
        if children:
            return children[-1]
        return fe
    elif "vit" in net_type or "factorized_vit" in net_type:
        return None
    return None


def save_attribution_nifti(
    attribution_np: np.ndarray, input_image: str, output_path: str
) -> None:
    """
    Save a numpy attribution volume as a NIfTI file.
    """
    attribution_np = np.transpose(attribution_np, (2, 1, 0))
    image = sitk.GetImageFromArray(attribution_np)
    image.CopyInformation(sitk.ReadImage(input_image))
    sitk.WriteImage(image, output_path)


def main(arguments):
    parser = Parser()

    parser.add_argument_by_key(
        [
            ("classification_net_type", "net_type"),
            "params_from",
            "dataset_json",
            "image_keys",
            "clinical_feature_keys",
            "adc_keys",
            "mask_key",
            "image_masking",
            "image_crop_from_mask",
            "n_classes",
            "filter_on_keys",
            "target_spacing",
            "pad_size",
            "crop_size",
            "subsample_size",
            "batch_size",
            "cache_rate",
            "config_file",
            "dev",
            "n_workers",
            "seed",
            "prediction_ids",
            ("prediction_checkpoints", "checkpoints"),
            "output_path",
        ]
    )

    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["gradcam", "integrated_gradients"],
        choices=["gradcam", "integrated_gradients"],
        help="Attribution methods to use.",
    )

    args = parser.parse_args(arguments)

    if args.params_from is not None:
        param_dict = get_params(args.params_from)
        args = merge_args(args, param_dict, sys.argv[1:])

    g, rng = get_generator_and_rng(args.seed)

    if args.clinical_feature_keys is None:
        clinical_feature_keys = []
    else:
        clinical_feature_keys = args.clinical_feature_keys

    data_dict = Dataset(args.dataset_json, rng=rng)
    presence_keys = args.image_keys + clinical_feature_keys
    if args.mask_key is not None:
        presence_keys.append(args.mask_key)
    data_dict.filter_dictionary(
        filters_presence=presence_keys,
        filters=args.filter_on_keys,
    )
    data_dict.subsample_dataset(subsample_size=args.subsample_size)

    if len(data_dict) == 0:
        raise Exception(
            "No data available for explanation "
            "(dataset={}; keys={})".format(args.dataset_json, args.image_keys)
        )

    keys = args.image_keys
    adc_keys = args.adc_keys if args.adc_keys is not None else []
    adc_keys = [k for k in adc_keys if k in keys]
    mask_key = args.mask_key
    input_keys = deepcopy(keys)
    if mask_key is not None:
        input_keys.append(mask_key)

    if args.net_type == "unet":
        network_config, _ = parse_config_unet(
            args.config_file, len(keys), args.n_classes
        )
    else:
        network_config = parse_config_cat(args.config_file)

    # used for integrated gradients
    if args.batch_size is not None:
        batch_size = args.batch_size
    elif "batch_size" in network_config:
        batch_size = network_config["batch_size"]
    else:
        batch_size = 4

    network_config["batch_size"] = 1

    transform_arguments = {
        "keys": keys,
        "mask_key": mask_key,
        "image_masking": args.image_masking,
        "image_crop_from_mask": args.image_crop_from_mask,
        "clinical_feature_keys": clinical_feature_keys,
        "adc_keys": adc_keys,
        "target_spacing": args.target_spacing,
        "crop_size": args.crop_size,
        "pad_size": args.pad_size,
    }

    transforms_prediction = ClassificationTransforms(
        **transform_arguments,
    ).transforms()

    invert_transform = monai.transforms.Invertd(
        keys=["explanation"],
        transform=transforms_prediction,
        orig_keys=["image"],
    )

    if args.n_classes == 2:
        network_config["loss_fn"] = torch.nn.BCEWithLogitsLoss()
    elif args.net_type == "ord":
        network_config["loss_fn"] = OrdinalSigmoidalLoss(
            n_classes=args.n_classes
        )
    else:
        network_config["loss_fn"] = torch.nn.CrossEntropyLoss()

    if args.prediction_ids:
        prediction_ids = parse_ids(args.prediction_ids)
    else:
        prediction_ids = [[k for k in data_dict]]

    output_dir = Path(args.output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    # PL sometimes needs a little hint to detect GPUs.
    torch.ones([1]).to("cuda" if "cuda" in args.dev else "cpu")

    for iteration in range(len(prediction_ids)):
        curr_prediction_ids = [
            pid for pid in prediction_ids[iteration] if pid in data_dict
        ]
        prediction_list = [data_dict[pid] for pid in curr_prediction_ids]

        prediction_dataset = monai.data.CacheDataset(
            prediction_list,
            transforms_prediction,
            num_workers=args.n_workers,
            cache_rate=args.cache_rate,
        )

        for checkpoint in args.checkpoints:
            logger.info("Loading checkpoint %s", checkpoint)
            network = get_classification_network(
                net_type=args.net_type,
                network_config=network_config,
                dropout_param=0,
                seed=None,
                n_classes=args.n_classes,
                keys=input_keys,
                clinical_feature_keys=clinical_feature_keys,
                train_loader_call=None,
                max_epochs=None,
                warmup_steps=None,
                start_decay=None,
                crop_size=args.crop_size,
                clinical_feature_means=None,
                clinical_feature_stds=None,
                label_smoothing=None,
                mixup_alpha=None,
                partial_mixup=None,
            )

            state_dict = torch.load(checkpoint, weights_only=False)[
                "state_dict"
            ]
            state_dict = {
                k: state_dict[k]
                for k in state_dict
                if "loss_fn.weight" not in k
            }
            network.load_state_dict(state_dict)
            network = network.eval().to(args.dev)

            is_ordinal = args.net_type == "ord"
            is_binary = args.n_classes == 2 and not is_ordinal

            if is_ordinal:
                wrapped_model = OrdinalRankWrapper(network)
            elif is_binary:
                wrapped_model = BinaryOutputWrapper(network)
            else:
                wrapped_model = network

            ckpt_name = Path(checkpoint).stem

            attr_methods = {}
            for method_name in args.methods:
                if method_name == "gradcam":
                    logger.info(f"Using gradcam")
                    last_conv = get_last_conv_layer(network, args.net_type)
                    if last_conv is None:
                        logger.warning(
                            "Could not find a convolutional layer for "
                            "GradCAM with net_type=%s, skipping GradCAM.",
                            args.net_type,
                        )
                        continue
                    attr_methods["gradcam"] = LayerGradCam(
                        wrapped_model, last_conv
                    )
                elif method_name == "integrated_gradients":
                    logger.info(
                        f"Using integrated gradients with batch size = {batch_size}"
                    )
                    attr_methods["integrated_gradients"] = IntegratedGradients(
                        wrapped_model
                    )

            if not attr_methods:
                logger.warning("No valid attribution methods, skipping.")
                continue

            with tqdm(total=len(curr_prediction_ids)) as pbar:
                for identifier, element in zip(
                    curr_prediction_ids, prediction_dataset
                ):
                    pbar.set_description("Explaining {}".format(identifier))
                    image = element["image"].unsqueeze(0).to(args.dev)
                    image.requires_grad_(True)

                    if is_ordinal:
                        target = None
                    elif is_binary:
                        with torch.no_grad():
                            logit = network(image)
                            pred_class = (
                                (torch.sigmoid(logit) > 0.5).long().item()
                            )
                        target = pred_class
                    else:
                        with torch.no_grad():
                            logits = network(image)
                            pred_class = logits.argmax(dim=-1).item()
                        target = pred_class

                    safe_id = str(identifier).replace("/", "_")
                    sample_dir = output_dir / safe_id / ckpt_name
                    sample_dir.mkdir(exist_ok=True, parents=True)

                    for attr_name, attr_method in attr_methods.items():
                        if attr_name == "gradcam":
                            attribution = attr_method.attribute(
                                image,
                                target=target,
                            )
                            attribution = LayerGradCam.interpolate(
                                attribution,
                                image.shape[2:],
                            )
                        elif attr_name == "integrated_gradients":
                            attribution = attr_method.attribute(
                                image,
                                target=target,
                                n_steps=50,
                                internal_batch_size=args.batch_size,
                            )
                            attribution = attribution.sum(dim=1, keepdim=True)

                        attribution = MetaTensor(
                            attribution[0],
                            meta=element["image"].meta.copy(),
                            applied_operations=element[
                                "image"
                            ].applied_operations.copy(),
                        )
                        with allow_missing_keys_mode(transforms_prediction):
                            attribution = transforms_prediction.inverse(
                                {"image": attribution}
                            )["image"]
                        attr_np = attribution.detach().cpu().numpy()

                        for ch_idx in range(attr_np.shape[0]):
                            fname = "{}_{}_ch{}.nii.gz".format(
                                attr_name, safe_id, ch_idx
                            )
                            save_attribution_nifti(
                                attr_np[ch_idx],
                                str(data_dict[identifier][keys[0]]),
                                sample_dir / fname,
                            )

                    pbar.update()

    logger.info("Explanations saved to %s", output_dir)

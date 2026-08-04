import json
import sys
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import monai
import numpy as np
import SimpleITK as sitk
import torch
from captum.attr import (
    DeepLift,
    GuidedBackprop,
    GuidedGradCam,
    IntegratedGradients,
    LayerGradCam,
)
from monai.data import MetaTensor
from monai.transforms.utils import allow_missing_keys_mode
from torch.nn.functional import conv3d
from tqdm import tqdm

from adell_mri.entrypoints.assemble_args import Parser
from adell_mri.entrypoints.classification._utils import (
    configure_loss_fn,
    create_classification_network,
    create_transform_arguments,
    extract_confounder_info,
    use_deconfounding,
)
from adell_mri.modules.config_parsing import parse_config_cat, parse_config_unet
from adell_mri.transform_factory.transforms import ClassificationTransforms
from adell_mri.utils.dataset import Dataset
from adell_mri.utils.parser import get_params, merge_args, parse_ids
from adell_mri.utils.python_logging import get_logger
from adell_mri.utils.torch_utils import (
    get_generator_and_rng,
    load_checkpoint_to_model,
)

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


class Attributions:
    def __init__(self, model: torch.nn.Module, net_type: str):
        self.model = model
        self.net_type = net_type
        self.attributions = {}

    def __getitem__(self, key):
        if key not in self.attributions:
            if key == "gradcam":
                logger.info("Using gradcam")
                last_conv = get_last_conv_layer(self.model, self.net_type)
                if last_conv is None:
                    raise ValueError(
                        "Could not find a convolutional layer "
                        f"for GradCAM with net_type={self.net_type}"
                    )
                self.attributions["gradcam"] = LayerGradCam(
                    self.model, last_conv
                )
            elif key == "integrated_gradients":
                logger.info("Using integrated gradients")
                self.attributions["integrated_gradients"] = IntegratedGradients(
                    self.model
                )
            elif key == "guided_gradcam":
                logger.info("Using guided gradcam")
                last_conv = get_last_conv_layer(self.model, self.net_type)
                if last_conv is None:
                    raise ValueError(
                        f"Could not find a convolutional layer "
                        f"for GuidedGradCAM with net_type={self.net_type}"
                    )
                self.attributions["guided_gradcam"] = GuidedGradCam(
                    self.model, last_conv
                )
            elif key == "deeplift":
                logger.info("Using deeplift")
                self.attributions["deeplift"] = DeepLift(self.model)
            elif key == "guided_backprop":
                logger.info("Using guided backprop")
                self.attributions["guided_backprop"] = GuidedBackprop(
                    self.model
                )
            else:
                raise ValueError(f"Unknown attribution method: {key}")
        return self.attributions[key]

    def __setitem__(self, key, value):
        self.attributions[key] = value

    def __len__(self):
        return len(self.attributions)

    def keys(self):
        return self.attributions.keys()

    def items(self):
        return self.attributions.items()


def apply_attribution(
    attr_name: str,
    attr_method,
    image: torch.Tensor,
    target: torch.Tensor | None,
    ig_n_steps: int = 25,
    batch_size: int = 4,
):
    """
    Wrapper for attribution.

    Args:
        attr_name: Name of the attribution method.
        attr_method: Attribution method.
        image: Input image.

    Returns:
        Attribution map.
    """
    if attr_name == "gradcam":
        attribution = attr_method.attribute(image, target=target)
        attribution = LayerGradCam.interpolate(attribution, image.shape[2:])
    elif attr_name == "integrated_gradients":
        attribution = attr_method.attribute(
            image,
            target=target,
            n_steps=ig_n_steps,
            internal_batch_size=batch_size,
        )
        attribution = attribution.sum(dim=1, keepdim=True)
    elif attr_name == "guided_gradcam":
        attribution = attr_method.attribute(image, target=target)
        attribution = attribution.sum(dim=1, keepdim=True)
    elif attr_name == "deeplift":
        baseline = gaussian_blur_3d(image[0], sigma=2.0).unsqueeze(0)
        attribution = attr_method.attribute(image, baseline, target=target)
        attribution = attribution.sum(dim=1, keepdim=True)
    elif attr_name == "guided_backprop":
        attribution = attr_method.attribute(image, target=target)
        attribution = attribution.sum(dim=1, keepdim=True)
    else:
        raise ValueError(f"Unknown attribution method: {attr_name}")
    return attribution


def gaussian_blur_3d(image, sigma=2.0, kernel_size=None):
    """
    Apply 3D Gaussian blur using PyTorch on GPU.

    Args:
        image: Tensor of shape (C, D, H, W) or (B, C, D, H, W)
        sigma: Standard deviation for Gaussian kernel
        kernel_size: Size of kernel (default: 2 * int(4 * sigma + 0.5) + 1)

    Returns:
        Blurred image tensor of same shape as input
    """
    if kernel_size is None:
        kernel_size = 2 * int(4 * sigma + 0.5) + 1

    ax = torch.arange(
        -kernel_size // 2 + 1.0,
        kernel_size // 2 + 1.0,
        device=image.device,
    )
    xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()

    if image.dim() == 4:
        c = image.shape[0]
        kernel = kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
        kernel = kernel.expand(c, 1, -1, -1, -1)
        padding = kernel_size // 2
        return conv3d(
            image.unsqueeze(0),
            kernel,
            padding=padding,
            groups=c,
        ).squeeze(0)
    else:  # (B, C, D, H, W)
        c = image.shape[1]
        kernel = kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
        kernel = kernel.expand(c, 1, -1, -1, -1)
        padding = kernel_size // 2
        return conv3d(image, kernel, padding=padding, groups=c)


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
    if hasattr(network, "model"):
        network = network.model
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


@lru_cache(maxsize=512)
def read_sitk(path: str):
    """
    Cached version of SimpleITK.ReadImage.

    Args:
        path: Path to the NIfTI file.

    Returns:
        SimpleITK image object.
    """
    return sitk.ReadImage(path)


def save_attribution_nifti(
    attribution_np: np.ndarray,
    input_image: str,
    output_path: str,
) -> None:
    """
    Save a numpy attribution volume as a NIfTI file.
    """
    attribution_np = np.transpose(attribution_np, (2, 1, 0))
    image = sitk.GetImageFromArray(attribution_np)
    image.CopyInformation(read_sitk(input_image))
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
            "cat_confounder_keys",
            "cont_confounder_keys",
            "exclude_surrogate_variables",
            "n_features_deconfounder",
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
        choices=[
            "gradcam",
            "integrated_gradients",
            "guided_gradcam",
            "deeplift",
            "guided_backprop",
        ],
        help="Attribution methods to use.",
    )
    parser.add_argument(
        "--ig_n_steps",
        type=int,
        default=25,
        help="Number of steps for Integrated Gradients (lower = faster).",
    )

    args = parser.parse_args(arguments)

    if args.params_from is not None:
        param_dict = get_params(args.params_from)
        args = merge_args(args, param_dict, sys.argv[1:])

    _g, rng = get_generator_and_rng(args.seed)

    if args.clinical_feature_keys is None:
        clinical_feature_keys = []
    else:
        clinical_feature_keys = args.clinical_feature_keys

    data_dict = Dataset(args.dataset_json, rng=rng)
    presence_keys = list(args.image_keys)
    if args.mask_key is not None:
        presence_keys.append(args.mask_key)
    if args.cat_confounder_keys is not None:
        presence_keys.extend(args.cat_confounder_keys)
    if args.cont_confounder_keys is not None:
        presence_keys.extend(args.cont_confounder_keys)
    if not use_deconfounding(args):
        presence_keys.extend(clinical_feature_keys)

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

    cat_key, cont_key, cat_vars, cont_vars = extract_confounder_info(
        args, data_dict
    )

    if args.net_type == "unet":
        network_config, _ = parse_config_unet(
            args.config_file, len(keys), args.n_classes
        )
    else:
        network_config = parse_config_cat(args.config_file)

    if args.batch_size is not None:
        batch_size = args.batch_size
    elif "batch_size" in network_config:
        batch_size = network_config["batch_size"]
    else:
        batch_size = 4

    network_config["batch_size"] = 1

    transform_arguments = create_transform_arguments(args)
    transforms_prediction = ClassificationTransforms(
        **transform_arguments,
    ).transforms()

    configure_loss_fn(
        network_config,
        None if use_deconfounding(args) else args.net_type,
        args.n_classes,
    )

    if args.prediction_ids:
        prediction_ids = parse_ids(args.prediction_ids)
    else:
        prediction_ids = [[k for k in data_dict]]

    output_dir = Path(args.output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    # PL sometimes needs a little hint to detect GPUs.
    torch.ones([1]).to("cuda" if "cuda" in args.dev else "cpu")

    global_output = []

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
            args.n_classes = args.n_classes
            network = create_classification_network(
                args=args,
                network_config=network_config,
                input_keys=input_keys,
                clinical_feature_keys=clinical_feature_keys,
                cat_vars=cat_vars,
                cont_vars=cont_vars,
                cat_key=cat_key,
                cont_key=cont_key,
                train_loader_call=None,
                max_epochs=None,
                warmup_steps=None,
                start_decay=None,
                clinical_feature_means=None,
                clinical_feature_stds=None,
            )

            load_checkpoint_to_model(
                network,
                checkpoint,
                exclude_from_state_dict=["loss_fn.weight"],
            )
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

            output_dict = {
                "iteration": iteration,
                "prediction_ids": curr_prediction_ids,
                "checkpoint": checkpoint,
                "predictions": {},
            }

            attr_methods = Attributions(wrapped_model, args.net_type)
            for method_name in args.methods:
                _ = attr_methods[method_name]

            if len(attr_methods) == 0:
                logger.warning("No valid attribution methods, skipping.")
                continue

            with tqdm(total=len(curr_prediction_ids)) as pbar:
                for identifier, element in zip(
                    curr_prediction_ids, prediction_dataset
                ):
                    safe_id = str(identifier).replace("/", "_")
                    sample_dir = output_dir / safe_id / ckpt_name
                    sample_dir.mkdir(exist_ok=True, parents=True)
                    file_exists = {
                        attr_name: len(
                            list(
                                sample_dir.glob(f"{attr_name}_{safe_id}*nii.gz")
                            )
                        )
                        > 0
                        for attr_name in attr_methods.keys()
                    }

                    network.zero_grad()
                    pbar.set_description(f"Explaining {identifier}")
                    image = element["image"].to(args.dev).unsqueeze(0)
                    if not all(file_exists.values()):
                        image.requires_grad_(True)

                    logits = network(image)
                    if is_ordinal:
                        target = None
                    elif is_binary:
                        pred_class = (torch.sigmoid(logits) > 0.5).long().item()
                        target = pred_class
                    else:
                        pred_class = logits.argmax(dim=-1).item()
                        target = pred_class

                    output_dict["predictions"][identifier] = (
                        logits.detach().cpu().numpy().tolist()
                    )

                    for attr_name, attr_method in attr_methods.items():
                        if file_exists[attr_name]:
                            continue
                        attribution = apply_attribution(
                            attr_name,
                            attr_method,
                            image,
                            target,
                            args.ig_n_steps,
                            batch_size,
                        )

                        attribution = MetaTensor(
                            attribution[0],
                            meta=element["image"].meta.copy(),
                            applied_operations=deepcopy(
                                element["image"].applied_operations
                            ),
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
                                output_path=sample_dir / fname,
                            )
                    pbar.update()

            global_output.append(output_dict)
    logger.info("Explanations saved to %s", output_dir)

    predictions_path = output_dir / "predictions.json"
    with open(predictions_path, "w") as o:
        o.write(json.dumps(global_output))

    logger.info("Predictions saved to %s", predictions_path)

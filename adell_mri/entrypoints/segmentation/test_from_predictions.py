import json
import os
import pprint
import re
from dataclasses import dataclass
from multiprocessing import Pool, Process, Queue
from pathlib import Path
from typing import Any, Callable

import numpy as np
import SimpleITK as sitk
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from tqdm import tqdm

from ...entrypoints.assemble_args import Parser
from ...modules.segmentation.picai_eval.eval import Metrics, evaluate_case
from ...utils.parser import parse_ids


def get_lesion_candidates(
    arr: np.ndarray,
    threshold: float,
    min_size: float = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shortened version of lesion candidate extraction.

    Args:
        arr (np.ndarray): probability array.
        threshold (float): threshold below which probability values are set to
            0.
        min_size (float): minimum size of lesion candidates. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: lesion candidates and lesion cancidates
            with all values above 0 set to the maximum probability of each
            lesion.
    """
    arr = np.where(arr < threshold, 0, arr)
    labels, max_label = ndimage.label(arr, structure=np.ones([3, 3, 3]))
    output = np.zeros_like(arr)
    output_eval = np.zeros_like(arr)
    for i in range(1, max_label + 1):
        if i != 0:
            if arr[labels == i].sum() > min_size:
                output[labels == i] = arr[labels == i]
                output_eval[labels == i] = arr[labels == i].max()
    return output, output_eval


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalises an array to be between 0 and 1.

    Args:
        image (np.ndarray): array.

    Returns:
        np.ndarray: normalised array.
    """
    m, M = image.min(), image.max()
    return (image - m) / (M - m)


def file_list_to_dict(file_list: list[str], pattern: str) -> dict[str, str]:
    """
    Converts a list of files to a dictionary of files where the key is
    extracted from the file name given a regex pattern.

    Args:
        file_list (list[str]): list of file paths.
        pattern (str): regex pattern for key.

    Returns:
        dict[str, str]: path dictionary.
    """
    pat = re.compile(pattern)
    output = {}
    for file in file_list:
        match = pat.search(file)
        if match is not None:
            match = match.group()
            output[match] = file
    return output


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid function.

    Args:
        x (np.ndarray): array.

    Returns:
        np.ndarray: array.
    """
    return 1 / (1 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Softmax function.

    Args:
        x (np.ndarray): array.
        axis (int, optional): axis along which softmax is calculated. Defaults
            to 0.

    Returns:
        np.ndarray: array.
    """
    x = np.exp(x)
    x = x / x.sum(axis, keepdims=True)
    return x


def coherce_to_serializable(d: dict) -> dict:
    """
    Recursive function to convert dictionaries to JSON serializable objects.

    Args:
        d (dict): dictionary structure.

    Returns:
        dict: dictionary with no numpy objects.
    """
    for k in d:
        if isinstance(d[k], np.ndarray):
            d[k] = d[k].tolist()
        if isinstance(d[k], (np.float32, np.float64, np.int32, np.int64)):
            d[k] = d[k].tolist()
        if isinstance(d[k], dict):
            d[k] = coherce_to_serializable(d[k])
    return d


def draw_legend(image: np.ndarray, cmap: Callable, mix_factor: float):
    """
    Draws a legend using a given matplotlib colour map.

    Args:
        image (np.ndarray): image where legend will be drawn.
        cmap (Callable): matplotlib colourmap.
        mix_factor (float): sets how blended the legend is with the image.
            Basically equivalent to alpha blending both, with 1.0 indicating
            only legend and 0.0 indicating no legend.
    """
    legend_coords = (image.shape[0] - 20, 20)
    legend_size = 50
    legend = cmap(np.linspace(0, 1, legend_size))
    legend *= np.minimum(mix_factor * 1.5, 1)
    vertical_size = 7
    for i in range(-(vertical_size // 2 + 1), (vertical_size // 2 + 2)):
        image[
            legend_coords[0] + i,
            (legend_coords[1] - 4) : (legend_coords[1] + legend_size + 4),
        ] = 1
    for i in range(-(vertical_size // 2 - 1), (vertical_size // 2)):
        image[
            legend_coords[0] + i,
            (legend_coords[1]) : (legend_coords[1] + legend_size),
        ] = legend


def visualize_heatmap(
    image: np.ndarray,
    heatmap_slice: np.ndarray,
    mix_factor: int = 1.0,
    do_legend: bool = True,
) -> np.ndarray:
    """
    Draws a heatmap on top of an image using the "jet" colourmap.

    Args:
        image (np.ndarray): RGBA image.
        heatmap_slice (np.ndarray): heatmap with values between 0 and 1.
        mix_factor (int, optional): mixture factor (works like alpha blending,
            1.0 is all heatmap, 0.0 is all image). Defaults to 1.0.
        do_legend (bool, optional): whether a legend should be drawn. Defaults
            to True.

    Returns:
        np.ndarray: image with a heatmap drawn on top.
    """
    if np.count_nonzero(heatmap_slice) > 0:
        cmap = colormaps["jet"]
        heatmap_slice = heatmap_slice
        heatmap_rgb = np.zeros((*heatmap_slice.shape, 4))
        x, y = np.where(heatmap_slice)
        heatmap_rgb[x, y] = cmap(heatmap_slice[x, y])
        heatmap_rgb[:, :, 3] = 1
        # R channel used for heatmap, G used for reverse
        image = np.where(
            heatmap_rgb > 0,
            heatmap_rgb * mix_factor + image * (1 - mix_factor),
            image,
        )
        if do_legend:
            draw_legend(image, cmap, mix_factor)

    return image


@dataclass
class ImageWriter:
    """
    Helper class that dispatches processes to write images to the disk.

    Args:
        n_proc (int, optional): number of processes to use. Defaults to 1.
    """

    n_proc: int = 1

    def __post_init__(self):
        self.queue = Queue()
        self.processes = [
            Process(target=self.write, args=(self.queue,))
            for _ in range(self.n_proc)
        ]
        for p in self.processes:
            p.start()

    def write(self, q):
        while True:
            image, image_name = q.get()
            if image_name is None:
                break
            image.save(image_name)

    def put(self, image, image_name):
        self.queue.put((image, image_name))

    # close queue and process
    def close(self):
        self.queue.put((None, None))
        for p in self.processes:
            p.join()
            p.close()


@dataclass
class CalculateMetrics:
    """
    Metrics calculator and example drawer. Metrics are calculated using the
    Radboud picai_eval library [1]. Examples are stacked images where the first
    rows represent the input images, the last row represents the
    predictions and the row before the last represents the ground truth. This
    is included for all slices with at least one positive prediction.

    For binary metrics (IoU, Dice, etc.) the `proba_threshold` has to be set
    as this is the threshold used to binarise images.

    This supports ensemble predictions as long as they are stacked on the first
    dimension. The `reduction` argument will be used to calculate the average
    value ("mean"), the maximum value ("max") or to extract a specific index of
    the ensemble (if a number is provided).

    This expects the input predictions to have 4 dimensions ((classes), h, w, d)
    and the input ground truth/examples to have 3 dimensions (h, w, d). This
    function only calculates binary metrics, so a class index has to be set
    through `class_idx`. If the number of dimensions is 3, it assumes that a
    class has been selected.

    Args:
        prediction_mode (str, optional): prediction mode. If "mask" does nothing
            to the prediction, if "logits" calculates the sigmoid/softmax
            depending on the number of classes, if "probs" does nothing to the
            prediction. Defaults to "mask".
        reduction (str, optional): reduction method. How ensemble values are
            reduced - this only happens if the dimension of the input is 5.
            Defaults to "mean".
        n_classes (int, optional): number of classes. Defaults to 2.
        overlap_threshold (float, optional): overlap threshold for detection
            metrics. Defaults to None.
        proba_threshold (float, optional): probability for binarising input.
        return_examples (bool, optional): whether example images are to be
            produced. Defaults to False.
        n_dim (int, optional): number of dimensions. Defaults to 3.
        class_idx (int, optional): index of class for metrics.
        min_size (float, optional): minimum size of candidates. Defaults to
            10.0.


    [1] https://github.com/DIAGNijmegen/picai_eval/tree/main
    """

    prediction_mode: str = "mask"
    reduction: str = "mean"
    n_classes: int = 2
    overlap_threshold: float = None
    proba_threshold: float = 0.1
    return_examples: bool = False
    n_dim: int = 3
    class_idx: int = 0
    min_size: float = 10.0

    def __post_init__(self):
        pass

    def read_image(self, image: str | sitk.Image | np.ndarray) -> np.ndarray:
        """
        Reads images in sitk.Image or np.ndarray format from a path allowing
        any intermediate format.

        Args:
            image (str | sitk.Image | np.ndarray): image to be read.

        Returns:
            np.ndarray: output array.
        """
        if isinstance(image, str):
            if image.endswith(".npz") or image.endswith(".npy"):
                image = np.load(image)
            else:
                image = sitk.ReadImage(image)
        if isinstance(image, sitk.Image):
            image = sitk.GetArrayFromImage(image)
        return image

    def preprocess_pred(self, pred: np.ndarray) -> np.ndarray:
        """
        Preprocesses prediction by converting it to the correct format and
        calculating the necessary ensemble values.

        Args:
            pred (np.ndarray): prediction array with either 4 or 5 dimensions.
                If ndim == 5 it assumes the first dimension corresponds to an
                ensemble dimension.

        Returns:
            np.ndarray: 4 dimensional array with class predictions.
        """
        if len(pred.shape) == 5:
            if self.reduction == "mean":
                pred = pred.mean(0)
            elif self.reduction == "sum":
                pred = pred.sum(0)
            elif self.reduction == "max":
                pred = pred.max(0)
            elif isinstance(self.reduction, int):
                pred = pred[self.reduction]
        if self.prediction_mode == "logits":
            if (pred.shape[0] == 1) or (len(pred.shape) == self.n_dim):
                pred = sigmoid(pred)
            else:
                pred = softmax(pred)
        return pred

    def extract_lesion_candidates(
        self, pred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Dumb wrapper around get_lesion_candidates.

        Args:
            pred (np.ndarray): probability array.

        Returns:
            tuple[np.ndarray, np.ndarray]: lesion candidates and lesion cancidates
                with all values above 0 set to the maximum probability of each
                lesion.
        """
        return get_lesion_candidates(pred, self.proba_threshold, self.min_size)

    def retrieve_example_image(
        self,
        image: np.ndarray,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
    ) -> np.ndarray:
        """
        Draws all slice examples from an image which have either a positive
        prediction or a positive pixel in the ground truth.

        Examples are stacked images where the first rows represent the input
        images, the last row represents the predictions and the row before the
        last represents the ground truth.

        Args:
            image (np.ndarray): input image.
            ground_truth (np.ndarray): ground truth binary array.
            prediction (np.ndarray): candidate map.

        Returns:
            np.ndarray: example image.
        """

        def stack_images_pil(*images: Image, axis: int = 0) -> Image:
            widths, heights = zip(*(i.size for i in images))
            if axis == 0:
                W = sum(widths)
                H = max(heights)
            else:
                W = max(widths)
                H = sum(heights)
            new_im = Image.new("RGBA", (W, H))
            offset = 0
            for im in images:
                if axis == 0:
                    new_im.paste(im, (offset, 0))
                else:
                    new_im.paste(im, (0, offset))
                offset += im.size[axis]
            return new_im

        z_gt, _, _ = np.where(ground_truth > 0.5)
        z_pred, _, _ = np.where(prediction)
        image = normalize(image)
        all_positive_slices = []
        all_positive_slices.extend(z_gt)
        all_positive_slices.extend(z_pred)
        all_positive_slices = np.unique(all_positive_slices)
        if len(all_positive_slices) == 0:
            return None
        output_figure = []
        first = True
        font = ImageFont.load_default(size=32)
        for idx in all_positive_slices:
            shaded = np.where(ground_truth[idx] > 0.5, 1, image[idx])
            shaded = np.stack([shaded for _ in range(4)], -1)
            shaded[:, :, -1] = 1
            image_stack = np.stack([image[idx] for _ in range(4)], -1)
            image_stack[:, :, -1] = 1
            if shaded.sum() == 0 and prediction[idx].sum == 0:
                continue
            heatmap = visualize_heatmap(
                image_stack, prediction[idx], do_legend=first
            )
            output_figure.append(
                stack_images_pil(
                    Image.fromarray(np.uint8(image_stack * 255)),
                    Image.fromarray(np.uint8(shaded * 255)),
                    Image.fromarray(np.uint8(heatmap * 255)),
                    axis=1,
                )
            )
            ImageDraw.Draw(output_figure[-1]).text(
                (5, 5), f"slice idx={idx}", (255, 255, 255), font=font
            )
            first = False
        output_figure = stack_images_pil(*output_figure, axis=0)
        return output_figure

    def overlap_dictionary(
        self, pred: np.ndarray, gt: np.ndarray
    ) -> dict[str, int]:
        """
        Calculates total positive pixels, intersection and union of two arrays.

        Args:
            pred (np.ndarray): binarized prediction array.
            gt (np.ndarray): ground truth array.

        Returns:
            dict[str, int]: dictionary with entries for total positive pixels
                in prediction and ground truth, and intersection and union
                between prediction and ground truth.
        """
        pred_sum = np.sum(pred == 1)
        gt_sum = np.sum(gt == 1)
        intersection = np.sum(np.logical_and(gt == pred, gt == 1))
        return {
            "pred_total": pred_sum,
            "gt_total": gt_sum,
            "intersection": intersection,
            "union": pred_sum + gt_sum - intersection,
        }

    def select_class(self, pred: np.ndarray):
        if len(pred.shape) == self.n_dim:
            return pred
        else:
            return pred[self.class_idx]

    def calculate_metrics(
        self,
        key: str,
        gt: str | sitk.Image,
        pred: str | sitk.Image,
        input_image: str | sitk.Image = None,
    ) -> tuple[str, dict[str, Any], Image.Image]:
        """
        Calculates metrics for a given key and generates example image if
        necessary.

        Args:
            key (str): key that will be returned with the output.
            gt (str | sitk.Image): ground truth.
            pred (str | sitk.Image): prediction probability map.
            input_image (str | sitk.Image, optional): input images for example
                image. Defaults to None.

        Returns:
            tuple[str, dict[str, Any], Image.Image]: key, output dict with all
                metrics and a PIL image with examples.
        """
        gt = self.read_image(gt)
        pred = self.read_image(pred)
        pred = self.select_class(self.preprocess_pred(pred))
        pred, pred_eval = self.extract_lesion_candidates(pred)
        y_list, case_confidence, _, _ = evaluate_case(
            y_det=pred_eval,
            y_true=gt,
            min_overlap=self.overlap_threshold,
            y_det_postprocess_func=None,
        )
        y_list = [
            {"gt": y[0], "confidence": y[1], "overlap": y[2]} for y in y_list
        ]
        gt_value = max([y["gt"] for y in y_list]) if len(y_list) > 0 else 0
        output_dict = {
            "lesions": y_list,
            "case_confidence": case_confidence,
            "gt": gt_value,
        }
        output_dict = {
            **output_dict,
            **self.overlap_dictionary(
                pred > self.proba_threshold, gt > self.proba_threshold
            ),
        }
        if input_image is not None and self.return_examples:
            input_image = self.read_image(input_image)
            input_image = self.retrieve_example_image(input_image, gt, pred)
        else:
            input_image = None
        return key, output_dict, input_image

    def calculate_metrics_wrapper(
        self,
        images: tuple[
            str, str | sitk.Image, str | sitk.Image, str | sitk.Image
        ],
    ):
        """
        Wrapper for self.calculate_metrics where the input is a tuple/list
        of all the inputs to self.calculate_metrics.
        """
        out = self.calculate_metrics(*images)
        return out


def main(arguments):
    parser = Parser(
        description="Calculates segmentation metrics from predictions and \
            ground truths."
    )

    parser.add_argument(
        "--ground_truth_path",
        required=True,
        help="Path to ground truth masks.",
    )
    parser.add_argument(
        "--ground_truth_patterns",
        default="*nii.gz",
        nargs="+",
        help="glob pattern which will be used to collect ground truths.",
    )
    parser.add_argument(
        "--prediction_path",
        required=True,
        help="Path to predictions",
    )
    parser.add_argument(
        "--prediction_patterns",
        default=["*nii.gz"],
        nargs="+",
        help="glob pattern which will be used to collect predictions.",
    )
    parser.add_argument(
        "--identifier_pattern",
        default="[0-9\\.]+\.[0-9\\.]+\.[0-9]+",
        help="Pattern for identifier",
    )
    parser.add_argument(
        "--prediction_mode",
        default="mask",
        choices=["mask", "probs", "logits"],
        help="Prediction mode. `mask` expects a categorical input with size \
            [h, w, d]. `logits` and `probs` expects input with shape \
            [h,w,d,n_classes,(batches)] and a candidate extraction algorithm is\
            applied to each case. `batches` refers to the number of ensemble\
            predictions. If `logits`, a softmax/sigmoid is applied if n_classes \
            is > 1 or == 1, respectively. If n_classes is 1, `logits` and `probs` \
            are assumed to be sigmoid probabilities and is converted to a 2 class \
            mask through concatenate([probs, 1 - probs]).",
    )
    parser.add_argument(
        "--label_groups",
        required=True,
        nargs="+",
        help="Space separated groups of labels. This is used to categorise \
            positive and negative cases in the ground truth and determine the \
            number of expected classes.",
    )
    parser.add_argument(
        "--reduction_mode",
        default="mean",
        help="If input has shape [h,w,d,n_classes,batches], the last dimension \
            will be reduced using this mode. If a number is specified, it will \
            extract that index.",
    )
    parser.add_argument(
        "--class_idx",
        default=0,
        help="Extracts this class from the data if more than one channel in \
            prediction",
    )
    parser.add_argument(
        "--n_dim",
        default=3,
        type=int,
        help="Number of dimensions",
    )
    parser.add_argument(
        "--overlap_threshold",
        default=0.1,
        type=float,
        help="IoU threshold to consider that an object has been detected",
    )
    parser.add_argument(
        "--proba_threshold",
        default=0.1,
        type=str,
        help="If a probability is > threshold then it is considered positive. \
            If not specified, assumes the maximum probability/0.5 in binary \
            cases corresponds to the correct/positive class",
    )
    parser.add_argument(
        "--min_size",
        default=10,
        type=str,
        help="Minimum prediction size",
    )
    parser.add_argument(
        "--id_list",
        default=None,
        nargs="+",
        help="List of ids in a format readable by `parse_ids`.",
    )
    parser.add_argument(
        "--n_workers",
        default=1,
        type=int,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        type=str,
        help="Output path. If not specified will print to stdout.",
    )
    parser.add_argument(
        "--generate_examples",
        action="store_true",
        help="Generates examples of predictions in PNG format. An example will \
            be generated for each study each having n slices, where n is the \
            number of slices.",
    )
    parser.add_argument(
        "--image_path",
        required=False,
        default=None,
        help="Path to images (for examples).",
    )
    parser.add_argument(
        "--image_patterns",
        default="*nii.gz",
        nargs="+",
        help="glob pattern which will be used to collect images (for examples).",
    )
    parser.add_argument(
        "--example_path",
        type=str,
        default="figures",
        help="Output path for examples.",
    )

    args = parser.parse_args(arguments)

    classes = [[int(y) for y in x.split(",")] for x in args.label_groups]
    n_classes = len(classes)

    all_ground_truth_paths = []
    for pattern in args.ground_truth_patterns:
        all_ground_truth_paths.extend(
            [str(x) for x in Path(args.ground_truth_path).glob(pattern)]
        )
    ground_truth_dict = file_list_to_dict(
        all_ground_truth_paths, args.identifier_pattern
    )
    print(f"Found ground truths: {len(all_ground_truth_paths)}")

    all_prediction_paths = []
    for pattern in args.prediction_patterns:
        all_prediction_paths.extend(
            [str(x) for x in Path(args.prediction_path).glob(pattern)]
        )
    prediction_dict = file_list_to_dict(
        all_prediction_paths, args.identifier_pattern
    )
    print(f"Found predictions: {len(all_prediction_paths)}")

    if args.generate_examples:
        all_image_paths = []
        for pattern in args.image_patterns:
            all_image_paths.extend(
                [str(x) for x in Path(args.image_path).glob(pattern)]
            )
        image_dict = file_list_to_dict(all_image_paths, args.identifier_pattern)
        print(f"Found example images: {len(image_dict)}")
    else:
        image_dict = {}

    if args.id_list is not None:
        id_list = parse_ids(args.id_list, "list")
    else:
        id_list = None

    merged_dict = {}
    for key in ground_truth_dict.keys():
        if key in prediction_dict:
            if id_list is not None:
                if key in id_list:
                    merged_dict[key] = {
                        "pred": prediction_dict[key],
                        "ground_truth": ground_truth_dict[key],
                        "example": image_dict.get(key, None),
                    }
            else:
                merged_dict[key] = {
                    "pred": prediction_dict[key],
                    "ground_truth": ground_truth_dict[key],
                    "example": image_dict.get(key, None),
                }

    print(f"Found matches: {len(merged_dict)}")

    if args.reduction_mode in ["mean", "sum", "max"]:
        reduction_mode = args.reduction_mode
    else:
        reduction_mode = int(args.reduction_mode)
    if args.proba_threshold not in ["dynamic", "dynamic-fast"]:
        args.proba_threshold = float(args.proba_threshold)
    metric_dict = CalculateMetrics(
        prediction_mode=args.prediction_mode,
        reduction=reduction_mode,
        n_classes=n_classes,
        overlap_threshold=args.overlap_threshold,
        proba_threshold=args.proba_threshold,
        return_examples=args.generate_examples,
        n_dim=args.n_dim,
        class_idx=args.class_idx,
        min_size=args.min_size,
    )

    for key in merged_dict:
        if key is None:
            print(key)
    input_list = [
        (
            key,
            merged_dict[key]["ground_truth"],
            merged_dict[key]["pred"],
            merged_dict[key]["example"],
        )
        for key in merged_dict
    ]
    if args.n_workers <= 1:
        iterator = map(metric_dict.calculate_metrics_wrapper, input_list)
    else:
        pool = Pool(args.n_workers)
        iterator = pool.imap(metric_dict.calculate_metrics_wrapper, input_list)

    all_outputs = {}
    if args.generate_examples:
        image_writer = ImageWriter()
    for key, output, example in tqdm(iterator, total=len(merged_dict)):
        all_outputs[key] = output
        if args.generate_examples and example is not None:
            Path(args.example_path).mkdir(exist_ok=True, parents=True)
            image_writer.put(
                example, os.path.join(args.example_path, key + ".png")
            )
    if args.generate_examples:
        print("Writing examples...")
        image_writer.close()

    lesion_results = {
        key: [
            [lesion[kk] for kk in ["gt", "confidence", "overlap"]]
            for lesion in all_outputs[key]["lesions"]
        ]
        for key in all_outputs
    }
    case_target = {key: all_outputs[key]["gt"] for key in all_outputs}
    case_pred = {
        key: all_outputs[key]["case_confidence"] for key in all_outputs
    }
    metrics = Metrics(
        lesion_results=lesion_results,
        case_target=case_target,
        case_pred=case_pred,
    )

    pr = metrics.calculate_precision_recall()
    roc = metrics.calculate_ROC()
    metric_dict = metrics.as_dict()
    metric_dict["lesion_results"] = {}
    for k in pr:
        metric_dict[k] = pr[k]
    for k in roc:
        metric_dict[k] = roc[k]
    for k in all_outputs:
        metric_dict["lesion_results"][k] = all_outputs[k]

    metric_dict = coherce_to_serializable(metric_dict)

    if args.output_json is not None:
        with open(args.output_json, "w") as o:
            json.dump(metric_dict, o, indent=2)
    else:
        pprint.pprint(metric_dict)

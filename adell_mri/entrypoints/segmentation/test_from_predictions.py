import pprint
import re
import json
import numpy as np
import SimpleITK as sitk
import os
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from scipy import ndimage
from multiprocessing import Pool, Queue, Process
from ...entrypoints.assemble_args import Parser
from ...utils.parser import parse_ids
from ...modules.segmentation.picai_eval.eval import evaluate_case, Metrics
from ...modules.extract_lesion_candidates import extract_lesion_candidates

from typing import Callable


def get_lesion_candidates(arr: np.ndarray, threshold: float) -> np.ndarray:
    arr = np.where(arr < threshold, 0, arr)
    labels, max_label = ndimage.label(arr, structure=np.ones([3, 3, 3]))
    output = np.zeros_like(arr)
    output_eval = np.zeros_like(arr)
    for i in range(1, max_label + 1):
        if i != 0:
            output[labels == i] = arr[labels == i]
            output_eval[labels == i] = arr[labels == i].max()
    return output, output_eval


def normalize(image: np.ndarray) -> np.ndarray:
    m, M = image.min(), image.max()
    return (image - m) / (M - m)


def file_list_to_dict(file_list: list[str], pattern: str) -> dict[str, str]:
    pat = re.compile(pattern)
    output = {}
    for file in file_list:
        match = pat.search(file)
        if match is not None:
            match = match.group()
            output[match] = file
    return output


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.exp(x)
    x = x / x.sum(0, keepdims=True)


def coherce_to_serializable(d: dict) -> dict:
    for k in d:
        if isinstance(d[k], np.ndarray):
            d[k] = d[k].tolist()
        if isinstance(d[k], (np.float32, np.float64)):
            d[k] = d[k].tolist()
        if isinstance(d[k], dict):
            d[k] = coherce_to_serializable(d[k])
    return d


def draw_legend(image_slice: np.ndarray, cmap: Callable, mix_factor: float):
    legend_coords = (image_slice.shape[0] - 20, 20)
    legend_size = 50
    legend = cmap(np.linspace(0, 1, legend_size))
    legend *= np.minimum(mix_factor * 1.5, 1)
    vertical_size = 7
    for i in range(-(vertical_size // 2 + 1), (vertical_size // 2 + 2)):
        image_slice[
            legend_coords[0] + i,
            (legend_coords[1] - 4) : (legend_coords[1] + legend_size + 4),
        ] = 1
    for i in range(-(vertical_size // 2 - 1), (vertical_size // 2)):
        image_slice[
            legend_coords[0] + i,
            (legend_coords[1]) : (legend_coords[1] + legend_size),
        ] = legend


def visualize_heatmap(
    image_slice: np.ndarray,
    heatmap_slice: np.ndarray,
    mix_factor: int = 1.0,
    do_legend: bool = True,
) -> np.ndarray:
    if np.count_nonzero(heatmap_slice) > 0:
        cmap = colormaps["jet"]
        heatmap_slice = heatmap_slice
        heatmap_rgb = np.zeros((*heatmap_slice.shape, 4))
        x, y = np.where(heatmap_slice)
        heatmap_rgb[x, y] = cmap(heatmap_slice[x, y])
        heatmap_rgb[:, :, 3] = 1
        # R channel used for heatmap, G used for reverse
        image_slice = np.where(
            heatmap_rgb > 0,
            heatmap_rgb * mix_factor + image_slice * (1 - mix_factor),
            image_slice,
        )
        if do_legend:
            draw_legend(image_slice, cmap, mix_factor)

    return image_slice


@dataclass
class ImageWriter:
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
    prediction_mode: str = "mask"
    reduction: str = "mean"
    n_classes: int = 2
    fold: int = None
    overlap_threshold: float = None
    proba_threshold: float = 0.1
    return_examples: bool = False
    class_idx: int = 0

    def __post_init__(self):
        pass

    def iou_dice_sizes(self, gt: np.ndarray, pred: np.ndarray):
        values = {}
        for i in np.unique(gt):
            is_gt = gt == i
            is_pred = pred == i
            gt_sum = is_gt.sum()
            pred_sum = is_pred.sum()
            intersection = np.logical_and(is_gt, is_pred).sum()
            values[i] = {
                "iou": intersection / (gt_sum + pred_sum - intersection),
                "dice": 2 * intersection / (gt_sum + pred_sum),
                "size_gt": gt_sum,
                "size_pred": pred_sum,
            }
        return values

    @property
    def metric_dict(self):
        return {"iou": self.iou_dice_sizes}

    def read_image(self, image: str | sitk.Image) -> np.ndarray:
        if isinstance(image, str):
            if image.endswith(".npz") or image.endswith(".npy"):
                image = np.load(image)
            else:
                image = sitk.ReadImage(image)
        if isinstance(image, sitk.Image):
            image = sitk.GetArrayFromImage(image)
        return image

    def pred_to_mask(self, pred: np.ndarray) -> np.ndarray:
        if self.prediction_mode in ["probs", "logits"]:
            if self.proba_threshold is None:
                if pred.shape[0] == 1:
                    pred = np.concatenate([1 - pred, pred])
                pred = np.argmax(pred, 0)
            else:
                pred = np.int32(pred > self.proba_threshold)
        return pred

    def preprocess_pred(self, pred: np.ndarray) -> np.ndarray:
        if len(pred.shape) == 5:
            if self.reduction == "mean":
                pred = pred.mean(0)
            elif self.reduction == "sum":
                pred = pred.sum(0)
            elif isinstance(self.reduction, int):
                pred = pred[self.reduction]
        if self.prediction_mode == "logits":
            if pred.shape[0] == 1:
                pred = sigmoid(pred)
            else:
                pred = softmax(pred)
        return pred

    def extract_lesion_candidates(self, pred: np.ndarray) -> np.ndarray:
        return get_lesion_candidates(pred, self.proba_threshold)

    def retrieve_example_image(
        self,
        image: np.ndarray,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
    ) -> np.ndarray:
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

    def calculate_metrics_wrapper(
        self,
        images: tuple[
            str, str | sitk.Image, str | sitk.Image, str | sitk.Image
        ],
    ):
        out = self.calculate_metrics(*images)
        return out

    def calculate_metrics(
        self,
        key: str,
        gt: str | sitk.Image,
        pred: str | sitk.Image,
        example: str | sitk.Image = None,
    ):
        gt = self.read_image(gt)
        pred = self.read_image(pred)
        pred = self.preprocess_pred(pred)[self.class_idx]
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
        if example is not None and self.return_examples:
            example = self.read_image(example)
            example = self.retrieve_example_image(example, gt, pred)
        else:
            example = None
        return key, output_dict, example


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
        default="*nii.gz",
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
        required=True,
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
        image_dict = file_list_to_dict(
            all_image_paths, args.identifier_pattern
        )
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

    if args.reduction_mode in ["mean", "sum"]:
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
            [lesion[k] for k in ["gt", "confidence", "overlap"]]
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

    for k in pr:
        metric_dict[k] = pr[k]
    for k in roc:
        metric_dict[k] = roc[k]
    metric_dict = coherce_to_serializable(metric_dict)

    if args.output_json is not None:
        with open(args.output_json, "w") as o:
            json.dump(metric_dict, o, indent=2)
    else:
        pprint.pprint(metric_dict)

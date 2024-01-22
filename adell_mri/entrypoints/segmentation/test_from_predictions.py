import re
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from multiprocessing import Pool

from ...entrypoints.assemble_args import Parser
from ...utils.parser import parse_ids
from ...modules.segmentation.picai_eval.eval import evaluate_case, Metrics
from ...modules.extract_lesion_candidates import extract_lesion_candidates


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


@dataclass
class CalculateMetrics:
    prediction_mode: str = "mask"
    reduction: str = "mean"
    n_classes: int = 2
    fold: int = None
    threshold: float = None
    proba_threshold: float = 0.1

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
            image = sitk.ReadImage(image)
        if isinstance(image, sitk.Image):
            image = sitk.GetArrayFromImage(image)
        return image

    def pred_to_mask(self, pred: np.ndarray) -> np.ndarray:
        if self.prediction_mode in ["probs", "logits"]:
            if self.threshold is None:
                if pred.shape[0] == 1:
                    pred = np.concatenate([1 - pred, pred])
                pred = np.argmax(pred, 0)
            else:
                pred = np.int32(pred > self.threshold)
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
        return extract_lesion_candidates(
            pred,
            threshold=self.proba_threshold,
        )[0]

    def calculate_metrics(
        self, images: tuple[str, str, str] | tuple[str, sitk.Image, sitk.Image]
    ):
        key, gt, pred = images
        gt = self.read_image(gt)
        pred = self.read_image(pred)
        pred = self.preprocess_pred(pred)
        y_list, case_confidence, _, _ = evaluate_case(
            y_det=pred[0],
            y_true=gt,
            min_overlap=self.threshold,
            y_det_postprocess_func=self.extract_lesion_candidates,
        )
        y_list = [
            {"gt": y[0], "confidence": y[1], "overlap": y[2]} for y in y_list
        ]
        gt = max([y["gt"] for y in y_list]) if len(y_list) > 0 else 0
        output_dict = {
            "lesions": y_list,
            "case_confidence": case_confidence,
            "gt": gt,
        }
        return key, output_dict


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
        "--threshold",
        default=None,
        type=float,
        help="IoU threshold to consider that an object has been detected",
    )
    parser.add_argument(
        "--proba_threshold",
        default=None,
        type=float,
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

    args = parser.parse_args(arguments)

    classes = [[int(y) for y in x.split(",")] for x in args.label_groups]
    n_classes = len(classes)

    all_ground_truth_paths = []
    for pattern in args.ground_truth_patterns:
        all_ground_truth_paths.extend(
            [str(x) for x in Path(args.ground_truth_path).glob(pattern)]
        )
    all_prediction_paths = []
    for pattern in args.prediction_patterns:
        all_prediction_paths.extend(
            [str(x) for x in Path(args.prediction_path).glob(pattern)]
        )

    print(f"Found ground truths: {len(all_ground_truth_paths)}")
    print(f"Found predictions: {len(all_prediction_paths)}")

    ground_truth_dict = file_list_to_dict(
        all_ground_truth_paths, args.identifier_pattern
    )

    prediction_dict = file_list_to_dict(
        all_prediction_paths, args.identifier_pattern
    )

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
                    }
            else:
                merged_dict[key] = {
                    "pred": prediction_dict[key],
                    "ground_truth": ground_truth_dict[key],
                }

    print(f"Found matches: {len(merged_dict)}")

    if args.reduction_mode in ["mean", "sum"]:
        reduction_mode = args.reduction_mode
    else:
        reduction_mode = int(args.reduction_mode)

    metric_dict = CalculateMetrics(
        prediction_mode=args.prediction_mode,
        reduction=reduction_mode,
        n_classes=n_classes,
        threshold=args.threshold,
    )

    for key in merged_dict:
        if key is None:
            print(key)
    input_list = [
        (key, merged_dict[key]["ground_truth"], merged_dict[key]["pred"])
        for key in merged_dict
    ]
    if args.n_workers <= 1:
        iterator = map(metric_dict.calculate_metrics, input_list)
    else:
        pool = Pool(args.n_workers)
        iterator = pool.imap(metric_dict.calculate_metrics, input_list)

    all_outputs = {}
    for key, output in tqdm(iterator, total=len(merged_dict)):
        all_outputs[key] = output

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
    print(
        Metrics(
            lesion_results=lesion_results,
            case_target=case_target,
            case_pred=case_pred,
        )
    )

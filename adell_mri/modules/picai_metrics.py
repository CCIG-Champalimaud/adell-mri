import concurrent.futures
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Path,
    PathLike,
    Sized,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from tqdm import tqdm

from adell_mri.utils.python_logging import get_logger

logger = get_logger(__name__)
label_structure = np.ones((3, 3, 3))


class Metrics:
    lesion_results: Union[
        Dict[Hashable, List[Tuple[int, float, float]]], PathLike
    ]
    case_target: Optional[Dict[Hashable, int]] = None
    case_pred: Optional[Dict[Hashable, float]] = None
    case_weight: Optional[Union[Dict[Hashable, float], List[float]]] = None
    lesion_weight: Optional[Dict[Hashable, List[float]]] = None
    thresholds: Optional[npt.NDArray[np.float64]] = None
    subject_list: Optional[List[str]] = None
    sort: bool = True

    def __post_init__(self):
        if isinstance(self.lesion_results, (str, Path)):
            # load metrics from file
            self.load(self.lesion_results)

        if self.subject_list is None:
            self.subject_list = sorted(list(self.lesion_results))

        if self.case_target is None:
            # derive case-level targets as the maximum lesion-level target
            self.case_target = {
                idx: (
                    max([is_lesion for is_lesion, _, _ in case_y_list])
                    if len(case_y_list)
                    else 0
                )
                for idx, case_y_list in self.lesion_results.items()
            }

        if self.case_pred is None:
            # derive case-level predictions as the maximum lesion-level prediction
            self.case_pred = {
                idx: (
                    max([confidence for _, confidence, _ in case_y_list])
                    if len(case_y_list)
                    else 0
                )
                for idx, case_y_list in self.lesion_results.items()
            }

        if not isinstance(self.case_weight, dict):
            subject_list = list(self.case_target)
            if self.case_weight is None:
                self.case_weight = {idx: 1 for idx in subject_list}
            else:
                self.case_weight = {
                    idx: weight
                    for idx, weight in zip(subject_list, self.case_weight)
                }

        if self.lesion_weight is None:
            subject_list = sorted(list(self.lesion_results))
            self.lesion_weight = {
                idx: [1] * len(case_y_list)
                for idx, case_y_list in self.lesion_results.items()
            }

        if self.sort:
            # sort dictionaries
            subject_list = sorted(list(self.lesion_results))
            self.lesion_results = {
                idx: self.lesion_results[idx] for idx in subject_list
            }
            self.lesion_weight = {
                idx: self.lesion_weight[idx] for idx in subject_list
            }
            self.case_target = {
                idx: self.case_target[idx] for idx in subject_list
            }
            self.case_pred = {idx: self.case_pred[idx] for idx in subject_list}
            self.case_weight = {
                idx: self.case_weight[idx] for idx in subject_list
            }

    # aggregates
    def calc_auroc(self, subject_list: Optional[List[str]] = None) -> float:
        """
        Calculate case-level Area Under the Receiver Operating Characteristic curve (AUROC)
        """
        return self.calculate_ROC(subject_list=subject_list)["AUROC"]

    @property
    def auroc(self) -> float:
        """
        Calculate case-level Area Under the Receiver Operating Characteristic curve (AUROC)
        """
        return self.calc_auroc()

    def calc_AP(self, subject_list: Optional[List[str]] = None) -> float:
        """
        Calculate Average Precision"""
        return self.calculate_precision_recall(subject_list=subject_list)["AP"]

    @property
    def AP(self) -> float:
        """
        Calculate Average Precision"""
        return self.calc_AP()

    @property
    def num_cases(self) -> int:
        """
        Calculate the number of cases"""
        return len(self.subject_list)

    @property
    def num_lesions(self) -> int:
        """
        Calculate the number of ground truth lesions"""
        return sum([is_lesion for is_lesion, *_ in self.lesion_results_flat])

    @property
    def score(self):
        """
        Calculate the ranking score, as used in the PI-CAI 22 Grand Challenge"""
        return (self.auroc + self.AP) / 2

    # lesion-level results
    def get_lesion_results_flat(self, subject_list: Optional[List[str]] = None):
        """
        Flatten the per-case lesion evaluation results into a single list"""
        if subject_list is None:
            subject_list = self.subject_list

        return [
            (is_lesion, confidence, overlap)
            for subject_id in subject_list
            for is_lesion, confidence, overlap in self.lesion_results[
                subject_id
            ]
        ]

    @property
    def lesion_results_flat(self) -> List[Tuple[int, float, float]]:
        """
        Flatten the per-case y_list"""
        return self.get_lesion_results_flat()

    def get_lesion_weight_flat(
        self, subject_list: Optional[List[str]] = None
    ) -> List[float]:
        """
        Retrieve lesion-wise sample weights (for a given subset of cases)"""
        if subject_list is None:
            subject_list = self.subject_list

        # collect lesion weights (and flatten)
        return [
            weight
            for subject_id in subject_list
            for weight in self.lesion_weight[subject_id]
        ]

    @property
    def lesion_weight_flat(self) -> List[float]:
        """
        Retrieve lesion-wise sample weights (for a given subset of cases)"""
        return self.get_lesion_weight_flat()

    @property
    def precision(self) -> "npt.NDArray[np.float64]":
        """
        Calculate lesion-level precision at each threshold"""
        return self.calculate_precision_recall()["precision"]

    @property
    def recall(self) -> "npt.NDArray[np.float64]":
        """
        Calculate lesion-level recall at each threshold"""
        return self.calculate_precision_recall()["recall"]

    @property
    def lesion_TP(self) -> "npt.NDArray[np.float64]":
        """
        Calculate number of true positive lesion detections at each threshold"""
        return self.calculate_counts()["TP"]

    @property
    def lesion_FP(self) -> "npt.NDArray[np.float64]":
        """
        Calculate number of false positive lesion detections at each threshold
        """
        return self.calculate_counts()["FP"]

    @property
    def lesion_TPR(self) -> "npt.NDArray[np.float64]":
        """
        Calculate lesion-level true positive rate (sensitivity) at each threshold
        """
        if self.num_lesions > 0:
            return self.lesion_TP / self.num_lesions
        else:
            return np.array([np.nan] * len(self.lesion_TP))

    @property
    def lesion_FPR(self) -> "npt.NDArray[np.float64]":
        """
        Calculate lesion-level false positive rate (number of false positives per case) at each threshold
        """
        return self.lesion_FP / self.num_cases

    # case-level results
    def calc_case_TPR(
        self, subject_list: Optional[List[str]] = None
    ) -> "npt.NDArray[np.float64]":
        """
        Calculate case-level true positive rate (sensitivity) at each threshold
        """
        return self.calculate_ROC(subject_list=subject_list)["TPR"]

    @property
    def case_TPR(self) -> "npt.NDArray[np.float64]":
        """
        Calculate case-level true positive rate (sensitivity) at each threshold
        """
        return self.calc_case_TPR()

    def calc_case_FPR(
        self, subject_list: Optional[List[str]] = None
    ) -> "npt.NDArray[np.float64]":
        """
        Calculate case-level false positive rate (1 - specificity) at each threshold
        """
        return self.calculate_ROC(subject_list=subject_list)["FPR"]

    @property
    def case_FPR(self) -> "npt.NDArray[np.float64]":
        """
        Calculate case-level false positive rate (1 - specificity) at each threshold
        """
        return self.calc_case_FPR()

    # supporting functions
    def calculate_counts(
        self, subject_list: Optional[List[str]] = None
    ) -> "Dict[str, npt.NDArray[np.float32]]":
        """
        Calculate lesion-level true positive (TP) detections and false positive (FP) detections as each threshold.
        """
        # flatten y_list (and select cases in subject_list)
        lesion_y_list = self.get_lesion_results_flat(subject_list=subject_list)

        # collect targets and predictions
        y_true: "npt.NDArray[np.float64]" = np.array(
            [target for target, *_ in lesion_y_list]
        )
        y_pred: "npt.NDArray[np.float64]" = np.array(
            [pred for _, pred, *_ in lesion_y_list]
        )

        if self.thresholds is None:
            # collect thresholds for lesion-based analysis
            self.thresholds = np.unique(y_pred)
            self.thresholds[
                ::-1
            ].sort()  # sort thresholds in descending order (inplace)

            # for >10,000 thresholds: resample to 10,000 unique thresholds, while also
            # keeping all thresholds higher than 0.8 and the first 20 thresholds
            if len(self.thresholds) > 10_000:
                rng = np.arange(
                    1,
                    len(self.thresholds),
                    len(self.thresholds) / 10_000,
                    dtype=np.int32,
                )
                st = [self.thresholds[i] for i in rng]
                low_thresholds = self.thresholds[-20:]
                self.thresholds = np.array(
                    [
                        t
                        for t in self.thresholds
                        if t > 0.8 or t in st or t in low_thresholds
                    ]
                )

        # define placeholders
        FP: "npt.NDArray[np.float32]" = np.zeros_like(
            self.thresholds, dtype=np.float32
        )
        TP: "npt.NDArray[np.float32]" = np.zeros_like(
            self.thresholds, dtype=np.float32
        )

        # for each threshold: count FPs and TPs
        for i, th in enumerate(self.thresholds):
            y_pred_thresholded = (y_pred >= th).astype(int)
            tp = np.sum(y_true * y_pred_thresholded)
            fp = np.sum(y_pred_thresholded - y_true * y_pred_thresholded)

            # update with new point
            FP[i] = fp
            TP[i] = tp

        # extend curve to infinity
        TP[-1] = TP[-2]
        FP[-1] = np.inf

        return {
            "TP": TP,
            "FP": FP,
        }

    def calculate_precision_recall(
        self, subject_list: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate Precision-Recall curve and calculate average precision (AP).
        """
        # flatten y_list (and select cases in subject_list)
        lesion_y_list = self.get_lesion_results_flat(subject_list=subject_list)

        # collect targets and predictions
        y_true: "npt.NDArray[np.float64]" = np.array(
            [target for target, *_ in lesion_y_list]
        )
        y_pred: "npt.NDArray[np.float64]" = np.array(
            [pred for _, pred, *_ in lesion_y_list]
        )

        # calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(
            y_true=y_true,
            probas_pred=y_pred,
            sample_weight=self.get_lesion_weight_flat(
                subject_list=subject_list
            ),
        )

        # set precision to zero at a threshold of "zero", as those lesion
        # candidates are included just to convey the number of lesions to
        # the precision_recall_curve function, and not actual candidates
        precision[:-1][thresholds == 0] = 0

        # calculate average precision using the step function integral
        # The following works because the last entry of precision is
        # guaranteed to be 1, as returned by precision_recall_curve
        # Taken from https://github.com/scikit-learn/scikit-learn/blob/
        # 32f9deaaf27c7ae56898222be9d820ba0fd1054f/sklearn/metrics/_ranking.py#L212
        AP = -np.sum(np.diff(recall) * np.array(precision)[:-1])

        return {
            "AP": AP,
            "precision": precision,
            "recall": recall,
        }

    def calculate_ROC(
        self, subject_list: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate Receiver Operating Characteristic curve for case-level risk stratification.
        """
        if subject_list is None:
            subject_list = self.subject_list

        fpr, tpr, _ = roc_curve(
            y_true=[self.case_target[s] for s in subject_list],
            y_score=[self.case_pred[s] for s in subject_list],
            sample_weight=[self.case_weight[s] for s in subject_list],
        )

        auroc = auc(fpr, tpr)

        return {
            "FPR": fpr,
            "TPR": tpr,
            "AUROC": auroc,
        }

    @property
    def version(self):
        return "1.4.x"

    def as_dict(self):
        return {
            # aggregates
            "auroc": self.auroc,
            "AP": self.AP,
            "num_cases": self.num_cases,
            "num_lesions": self.num_lesions,
            "picai_eval_version": self.version,
            # lesion-level results
            "lesion_results": self.lesion_results,
            "lesion_weight": self.lesion_weight,
            # case-level results
            "case_pred": self.case_pred,
            "case_target": self.case_target,
            "case_weight": self.case_weight,
        }

    def full_dict(self):
        return {
            # aggregates
            "auroc": self.auroc,
            "AP": self.AP,
            "num_cases": self.num_cases,
            "num_lesions": self.num_lesions,
            "picai_eval_version": self.version,
            # lesion-level results
            "lesion_results": self.lesion_results,
            "lesion_weight": self.lesion_weight,
            "precision": self.precision,
            "recall": self.recall,
            "lesion_TPR": self.lesion_TPR,
            "lesion_FPR": self.lesion_FPR,
            "thresholds": self.thresholds,
            # case-level results
            "case_pred": self.case_pred,
            "case_target": self.case_target,
            "case_weight": self.case_weight,
        }

    def minimal_dict(self):
        return {
            # lesion-level results
            "lesion_results": self.lesion_results,
            "lesion_weight": self.lesion_weight,
            # case-level results
            "case_pred": self.case_pred,
            "case_target": self.case_target,
            "case_weight": self.case_weight,
        }

    def load_metrics(self, file_path: PathLike):
        """
        Read metrics from disk"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Metrics not found at {file_path}!")

        with open(file_path) as fp:
            metrics = json.load(fp)

        return metrics

    def load(self, path: PathLike):
        """
        Load metrics from file"""
        metrics = self.load_metrics(path)

        # parse metrics
        self.case_target = {
            idx: int(float(val)) for idx, val in metrics["case_target"].items()
        }
        self.case_pred = {
            idx: float(val) for idx, val in metrics["case_pred"].items()
        }
        self.case_weight = {
            idx: float(val) for idx, val in metrics["case_weight"].items()
        }
        self.lesion_weight = {
            idx: [float(val) for val in weights]
            for idx, weights in metrics["lesion_weight"].items()
        }
        self.lesion_results = {
            idx: [
                (int(float(is_lesion)), float(confidence), float(overlap))
                for (is_lesion, confidence, overlap) in lesion_results_case
            ]
            for idx, lesion_results_case in metrics["lesion_results"].items()
        }

    def __str__(self) -> str:
        return f"Metrics(auroc={self.auroc:.2%}, AP={self.AP:.2%}, {self.num_cases} cases, {self.num_lesions} lesions)"

    def __repr__(self) -> str:
        return self.__str__()


def read_label(path: PathLike) -> "npt.NDArray[np.int32]":
    """
    Read label, given a filepath.
    """
    # read label and ensure correct dtype
    lbl: "npt.NDArray[np.int32]" = np.array(read_image(path), dtype=np.int32)
    return lbl


def read_image(path: PathLike):
    """
    Read image, given a filepath.
    """
    if isinstance(path, Path):
        path = path.as_posix()
    else:
        assert isinstance(
            path, str
        ), f"Unexpected path type: {type(path)}. Please provide a Path or str."

    if ".npy" in path:
        return np.load(path)
    elif ".nii" in path or ".mha" in path or "mhd" in path:
        return sitk.GetArrayFromImage(sitk.ReadImage(path))
    elif ".npz" in path:
        return np.load(path)["softmax"].astype("float32")[1]  # nnUnet format
    else:
        raise ValueError(
            f"Unexpected file path. Supported file formats: .nii(.gz), .mha, .npy and .npz. Got: {path}."
        )


def calculate_dsc(
    y_det: "npt.NDArray[np.float32]", y_true: "npt.NDArray[np.int32]"
) -> float:
    """
    Calculate Dice similarity coefficient (DSC) for N-D Arrays.
    """
    epsilon = 1e-8
    dsc_num = np.sum(y_det[y_true == 1]) * 2.0
    dsc_denom = np.sum(y_det) + np.sum(y_true)
    return float((dsc_num + epsilon) / (dsc_denom + epsilon))


def parse_detection_map(
    y_det: "npt.NDArray[np.float32]",
) -> "Tuple[Dict[int, float], npt.NDArray[np.int32]]":
    """
    Extract confidence scores per lesion candidate.
    """
    # label all non-connected components in the detection map
    blobs_index, num_blobs = ndimage.label(y_det, structure=label_structure)

    # input verification
    if num_blobs < len(set(np.unique(y_det)) - {0}):
        raise ValueError(
            "It looks like the provided detection map is a softmax volume. If this is indeed the case, convert "
            "the softmax volumes to detection maps. Check the documentation how to incorporate this: "
            "https://github.com/DIAGNijmegen/picai_eval/."
        )

    # extract confidence per lesion candidate
    confidences = {}
    for lesion_candidate_id in range(num_blobs):
        max_prob = y_det[blobs_index == (1 + lesion_candidate_id)].max()
        confidences[lesion_candidate_id] = float(max_prob)

    return confidences, blobs_index


def calculate_iou(
    y_det: "npt.NDArray[np.float32]", y_true: "npt.NDArray[np.int32]"
) -> float:
    """
    Calculate Intersection over Union (IoU) for N-D Arrays.
    """
    epsilon = 1e-8
    iou_num = np.sum(y_det[y_true == 1])
    iou_denom = np.sum(y_det) + np.sum(y_true) - iou_num
    return float((iou_num + epsilon) / (iou_denom + epsilon))


def read_prediction(path: PathLike) -> "npt.NDArray[np.float32]":
    """
    Read prediction, given a filepath.
    """
    # read prediction and ensure correct dtype
    pred: "npt.NDArray[np.float32]" = np.array(
        read_image(path), dtype=np.float32
    )
    return pred


def resize_image_with_crop_or_pad(image, img_size=(64, 64, 64), **kwargs):
    """
    Resize images (scans/predictions/labels) by cropping and/or padding
    Adapted from: https://github.com/DLTK/DLTK]
    """
    assert isinstance(image, np.ndarray)
    assert image.ndim - 1 == len(img_size) or image.ndim == len(
        img_size
    ), "Target size doesn't fit image size"

    rank = len(img_size)  # image dimensions

    # placeholders for new shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for _ in range(rank)]
    slicer = [slice(None)] * rank

    # for each dimension, determine process (cropping or padding)
    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(
                np.floor((image.shape[i] - img_size[i]) / 2.0)
            )
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        # create slicer object to crop/leave each dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    # pad cropped image to extend missing dimension
    return np.pad(image[tuple(slicer)], to_padding, **kwargs)


def evaluate_case(
    y_det: "Union[npt.NDArray[np.float32], str, Path]",
    y_true: "Union[npt.NDArray[np.int32], str, Path]",
    min_overlap: float = 0.10,
    overlap_func: "Union[str, Callable[[npt.NDArray[np.float32], npt.NDArray[np.int32]], float]]" = "IoU",
    case_confidence_func: "Union[str, Callable[[npt.NDArray[np.float32]], float]]" = "max",
    allow_unmatched_candidates_with_minimal_overlap: bool = True,
    y_det_postprocess_func: "Optional[Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]]" = None,
    y_true_postprocess_func: "Optional[Callable[[npt.NDArray[np.int32]], npt.NDArray[np.int32]]]" = None,
) -> Tuple[List[Tuple[int, float, float]], float]:
    """
    Gather the list of lesion candidates, and classify in TP/FP/FN.
    Lesion candidates are matched to ground truth lesions, by maximizing the number of candidates
    with sufficient overlap (i.e., matches), and secondly by maximizing the total overlap of all candidates.
    Parameters:
    - y_det: Detection map, which should be a 3D volume containing connected components (in 3D) of the
        same confidence. Each detection map may contain an arbitrary number of connected components,
        with different or equal confidences. Alternatively, y_det may be a filename ending in
        .nii.gz/.mha/.mhd/.npy/.npz, which will be loaded on-the-fly.
    - y_true: Ground truth label, which should be a 3D volume of the same shape as the detection map.
        Alternatively, `y_true` may be the filename ending in .nii.gz/.mha/.mhd/.npy/.npz, which should
        contain binary labels and will be loaded on-the-fly. Use `1` to encode ground truth lesion, and
        `0` to encode background.
    - min_overlap: defines the minimal required overlap (e.g., Intersection over Union or Dice similarity
        coefficient) between a lesion candidate and ground truth lesion, to be counted as a true positive
        detection.
    - overlap_func: function to calculate overlap between a lesion candidate and ground truth mask.
        May be 'IoU' for Intersection over Union, or 'DSC' for Dice similarity coefficient. Alternatively,
        provide a function with signature `func(detection_map, annotation) -> overlap [0, 1]`.
    - allow_unmatched_candidates_with_minimal_overlap: when multiple lesion candidates have sufficient
        overlap with the ground truth lesion mask, this determines whether the lesion that is not selected
        counts as a false positive.
    - y_det_postprocess_func: function to apply to detection map. Can for example be used to extract
        lesion candidates from a softmax prediction volume.
    - y_true_postprocess_func: function to apply to annotation. Can for example be used to select the lesion
        masks from annotations that also contain other structures (such as organ segmentations).
    Returns:
    - a list of tuples with:
        (is_lesion, prediction confidence, overlap)
    - case level confidence score derived from the detection map
    """
    y_list: List[Tuple[int, float, float]] = []
    if isinstance(y_true, (str, Path)):
        y_true = read_label(y_true)
    if isinstance(y_det, (str, Path)):
        y_det = read_prediction(y_det)
    if overlap_func == "IoU":
        overlap_func = calculate_iou
    elif overlap_func == "DSC":
        overlap_func = calculate_dsc
    elif isinstance(overlap_func, str):
        raise ValueError(
            f"Overlap function with name {overlap_func} not recognized. Supported are 'IoU' and 'DSC'"
        )

    # convert dtype to float32
    y_true = y_true.astype("int32")
    y_det = y_det.astype("float32")

    # if specified, apply postprocessing functions
    if y_det_postprocess_func is not None:
        y_det = y_det_postprocess_func(y_det)
    if y_true_postprocess_func is not None:
        y_true = y_true_postprocess_func(y_true)

    # check if detection maps need to be padded
    if y_det.shape[0] < y_true.shape[0]:
        logger.info("Warning: padding prediction to match label!")
        y_det = resize_image_with_crop_or_pad(y_det, y_true.shape)
    if np.min(y_det) < 0:
        raise ValueError("All detection confidences must be positive!")

    # perform connected-components analysis on detection maps
    confidences, indexed_pred = parse_detection_map(y_det)
    lesion_candidate_ids = np.arange(len(confidences))

    if not y_true.any():
        # benign case, all predictions are FPs
        for lesion_confidence in confidences.values():
            y_list.append((0, lesion_confidence, 0.0))
    else:
        # malignant case, collect overlap between each prediction and ground truth lesion
        labeled_gt, num_gt_lesions = ndimage.label(
            y_true, structure=label_structure
        )
        gt_lesion_ids = np.arange(num_gt_lesions)
        overlap_matrix = np.zeros((num_gt_lesions, len(confidences)))

        for lesion_id in gt_lesion_ids:
            # for each lesion in ground-truth (GT) label
            gt_lesion_mask = labeled_gt == (1 + lesion_id)

            # calculate overlap between each lesion candidate and the current GT lesion
            for lesion_candidate_id in lesion_candidate_ids:
                # calculate overlap between lesion candidate and GT mask
                lesion_pred_mask = indexed_pred == (1 + lesion_candidate_id)
                overlap_score = overlap_func(lesion_pred_mask, gt_lesion_mask)

                # store overlap
                overlap_matrix[lesion_id, lesion_candidate_id] = overlap_score

        # match lesion candidates to ground truth lesion (for documentation on how this works, please see
        # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html)
        overlap_matrix[
            overlap_matrix < min_overlap
        ] = 0  # don't match lesions with insufficient overlap
        overlap_matrix[
            overlap_matrix > 0
        ] += 1  # prioritize matching over the amount of overlap
        (
            matched_lesion_indices,
            matched_lesion_candidate_indices,
        ) = linear_sum_assignment(overlap_matrix, maximize=True)

        # remove indices where overlap is zero
        mask = (
            overlap_matrix[
                matched_lesion_indices, matched_lesion_candidate_indices
            ]
            > 0
        )
        matched_lesion_indices = matched_lesion_indices[mask]
        matched_lesion_candidate_indices = matched_lesion_candidate_indices[
            mask
        ]

        # all lesion candidates that are matched are TPs
        for lesion_id, lesion_candidate_id in zip(
            matched_lesion_indices, matched_lesion_candidate_indices
        ):
            lesion_confidence = confidences[lesion_candidate_id]
            overlap = overlap_matrix[lesion_id, lesion_candidate_id]
            overlap -= 1  # return overlap to [0, 1]

            assert (
                overlap > min_overlap
            ), "Overlap must be greater than min_overlap!"

            y_list.append((1, lesion_confidence, overlap))

        # all ground truth lesions that are not matched are FNs
        unmatched_gt_lesions = set(gt_lesion_ids) - set(matched_lesion_indices)
        y_list += [(1, 0.0, 0.0) for _ in unmatched_gt_lesions]

        # all lesion candidates with insufficient overlap/not matched to a gt lesion are FPs
        if allow_unmatched_candidates_with_minimal_overlap:
            candidates_sufficient_overlap = lesion_candidate_ids[
                (overlap_matrix > 0).any(axis=0)
            ]
            unmatched_candidates = set(lesion_candidate_ids) - set(
                candidates_sufficient_overlap
            )
        else:
            unmatched_candidates = set(lesion_candidate_ids) - set(
                matched_lesion_candidate_indices
            )
        y_list += [
            (0, confidences[lesion_candidate_id], 0.0)
            for lesion_candidate_id in unmatched_candidates
        ]

    # determine case-level confidence score
    if case_confidence_func == "max":
        # take highest lesion confidence as case-level confidence
        case_confidence = np.max(y_det)
    elif case_confidence_func == "bayesian":
        # if c_i is the probability the i-th lesion is csPCa, then the case-level
        # probability to have one or multiple csPCa lesion is 1 - Π_i{ 1 - c_i}
        case_confidence = 1 - np.prod([(1 - c) for c in confidences.values()])
    else:
        # apply user-defines case-level confidence score function
        case_confidence = case_confidence_func(y_det)

    return y_list, case_confidence


# Evaluate all cases
def evaluate(
    y_det: "Iterable[Union[npt.NDArray[np.float64], str, Path]]",
    y_true: "Iterable[Union[npt.NDArray[np.float64], str, Path]]",
    sample_weight: "Optional[Iterable[float]]" = None,
    subject_list: Optional[Iterable[Hashable]] = None,
    min_overlap: float = 0.10,
    overlap_func: "Union[str, Callable[[npt.NDArray[np.float32], npt.NDArray[np.int32]], float]]" = "IoU",
    case_confidence_func: "Union[str, Callable[[npt.NDArray[np.float32]], float]]" = "max",
    allow_unmatched_candidates_with_minimal_overlap: bool = True,
    y_det_postprocess_func: "Optional[Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]]" = None,
    y_true_postprocess_func: "Optional[Callable[[npt.NDArray[np.int32]], npt.NDArray[np.int32]]]" = None,
    num_parallel_calls: int = 3,
    verbose: int = 0,
) -> Metrics:
    """
    Evaluate 3D detection performance.
    Parameters:
    - y_det: iterable of all detection_map volumes to evaluate. Each detection map should a 3D volume
        containing connected components (in 3D) of the same confidence. Each detection map may contain
        an arbitrary number of connected components, with different or equal confidences.
        Alternatively, y_det may contain filenames ending in .nii.gz/.mha/.mhd/.npy/.npz, which will
        be loaded on-the-fly.
    - y_true: iterable of all ground truth labels. Each label should be a 3D volume of the same shape
        as the corresponding detection map. Alternatively, `y_true` may contain filenames ending in
        .nii.gz/.mha/.mhd/.npy/.npz, which should contain binary labels and will be loaded on-the-fly.
        Use `1` to encode ground truth lesion, and `0` to encode background.
    - sample_weight: case-level sample weight. These weights will also be applied to the lesion-level
        evaluation, with same weight for all lesion candidates of the same case.
    - subject_list: list of sample identifiers, to give recognizable names to the evaluation results.
    - min_overlap: defines the minimal required Intersection over Union (IoU) or Dice similarity
        coefficient (DSC) between a lesion candidate and ground truth lesion, to be counted as a true
        positive detection.
    - overlap_func: function to calculate overlap between a lesion candidate and ground truth mask.
        May be 'IoU' for Intersection over Union, or 'DSC' for Dice similarity coefficient. Alternatively,
        provide a function with signature `func(detection_map, annotation) -> overlap [0, 1]`.
    - case_confidence_func: function to derive case-level confidence from detection map. Default: max.
    - allow_unmatched_candidates_with_minimal_overlap: when multiple lesion candidates have sufficient
        overlap with the ground truth lesion mask, this determines whether the lesion that is not selected
        counts as a false positive.
    - y_det_postprocess_func: function to apply to detection map. Can for example be used to extract
        lesion candidates from a softmax prediction volume.
    - y_true_postprocess_func: function to apply to annotation. Can for example be used to select the lesion
        masks from annotations that also contain other structures (such as organ segmentations).
    - num_parallel_calls: number of threads to use for evaluation.
    - verbose: (optional) controll amount of printed information.
    Returns:
    - Metrics
    """
    if sample_weight is None:
        sample_weight = itertools.repeat(1)
    if subject_list is None:
        # generate indices to keep track of each case during multiprocessing
        subject_list = itertools.count()

    # initialize placeholders
    case_target: Dict[Hashable, int] = {}
    case_weight: Dict[Hashable, float] = {}
    case_pred: Dict[Hashable, float] = {}
    lesion_results: Dict[Hashable, List[Tuple[int, float, float]]] = {}
    lesion_weight: Dict[Hashable, List[float]] = {}

    with ThreadPoolExecutor(max_workers=num_parallel_calls) as pool:
        # define the functions that need to be processed: compute_pred_vector, with each individual
        # detection_map prediction, ground truth label and parameters
        future_to_args = {
            pool.submit(
                evaluate_case,
                y_det=y_det_case,
                y_true=y_true_case,
                min_overlap=min_overlap,
                overlap_func=overlap_func,
                case_confidence_func=case_confidence_func,
                allow_unmatched_candidates_with_minimal_overlap=allow_unmatched_candidates_with_minimal_overlap,
                y_det_postprocess_func=y_det_postprocess_func,
                y_true_postprocess_func=y_true_postprocess_func,
            ): (idx, weight)
            for (y_det_case, y_true_case, weight, idx) in zip(
                y_det, y_true, sample_weight, subject_list
            )
        }

        # process the cases in parallel
        iterator = concurrent.futures.as_completed(future_to_args)
        if verbose:
            total: Optional[int] = None
            if isinstance(subject_list, Sized):
                total = len(subject_list)
            iterator = tqdm(iterator, desc="Evaluating", total=total)

        for future in iterator:
            idx, weight = future_to_args[future]

            try:
                # unpack results
                lesion_results_case, case_confidence = future.result()
            except Exception as e:
                logger.error("Error for %s: %s", idx, e)
                raise e

            # aggregate results
            idx, weight = future_to_args[future]
            case_weight[idx] = weight
            case_pred[idx] = case_confidence
            if len(lesion_results_case):
                case_target[idx] = np.max([a[0] for a in lesion_results_case])
            else:
                case_target[idx] = 0

            # accumulate outputs
            lesion_results[idx] = lesion_results_case
            lesion_weight[idx] = [weight] * len(lesion_results_case)

    # collect results in a Metrics object
    metrics = Metrics(
        lesion_results=lesion_results,
        case_target=case_target,
        case_pred=case_pred,
        case_weight=case_weight,
        lesion_weight=lesion_weight,
    )

    return metrics

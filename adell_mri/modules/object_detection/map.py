import numpy as np
import torch
import torchmetrics

from adell_mri.modules.object_detection.utils import (
    calculate_iou,
    check_overlap,
)


class mAP(torchmetrics.metric.Metric):
    def __init__(
        self, ndim=3, score_threshold=0.5, iou_threshold=0.5, n_classes=2
    ):
        """
        Mean average precision implementation for any number of dimensions.

        Args:
            ndim (int, optional): number of dimensions. Defaults to 3.
            score_threshold (float, optional): objectness score threshold.
                Defaults to 0.5.
            iou_threshold (float, optional): IoU threshold. Defaults to 0.5.
            n_classes (int, optional): number of classes. Defaults to 2.
        """
        super().__init__()
        self.ndim = ndim
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.n_classes = n_classes
        # will do most of the AP heavylifting
        nc = None if n_classes == 2 else n_classes
        self.average_precision = torchmetrics.AveragePrecision(
            task="binary" if nc is None else "multiclass", num_classes=nc
        )

        self.pred_list = []
        self.target_list = []
        self.pred_keys = ["boxes", "scores", "labels"]
        self.target_keys = ["boxes", "labels"]

        self.hits = 0

    def check_input(self, x, what):
        if what == "pred":
            K = self.pred_keys
        elif what == "target":
            K = self.target_keys
        for y in x:
            keys = y.keys()
            if all([k in keys for k in K]):
                pass
            else:
                raise ValueError("{} should have the keys {}".format(what, K))
            for k in K:
                if len(y[k].shape) == 0:
                    y[k] = y[k].unsqueeze(0)
            if len(np.unique([y[k].shape[0] for k in K])) > 1:
                raise ValueError(
                    "Inputs in {} should have the same number of elements".format(
                        what
                    )
                )

    def update(self, pred, target):
        self.check_input(pred, "pred")
        self.check_input(target, "target")

        self.pred_list.extend(pred)
        self.target_list.extend(target)

    def forward(self, pred, target):
        self.update(pred, target)

    def compute_image(self, pred, target):
        pbb, ps, pcp = [pred[k] for k in self.pred_keys]
        tbb, tc = [target[k] for k in self.target_keys]

        # step 1 - exclude low confidence predictions, sort by scores
        ps_v = ps > self.score_threshold
        pbb, ps, pcp = pbb[ps_v], ps[ps_v], pcp[ps_v]
        score_order = torch.argsort(ps)
        pbb = pbb[score_order]
        ps = ps[score_order]
        pcp = pcp[score_order]

        # step 2 - calculate iou
        n_pred = pbb.shape[0]
        n_target = tbb.shape[0]
        iou_array = torch.zeros([n_target, n_pred], device=pcp.device)
        any_hit = False
        for i in range(tbb.shape[0]):
            cur_bb = torch.unsqueeze(tbb[i], 0)
            # start by calculating overlap
            overlap = check_overlap(cur_bb, pbb, self.ndim)
            if overlap.sum() > 0:
                cur_pbb = pbb[overlap]
                iou = calculate_iou(cur_bb, cur_pbb, self.ndim)
                iou_array[i, overlap] = iou.float()
                any_hit = True

        if any_hit is True:
            # step 3 - filter by highest iou
            best_pred = torch.argmax(iou_array, 1)

            # step 4 - threshold highest iou
            true_ious = iou_array[torch.arange(0, n_target), best_pred]
            hit = true_ious > self.iou_threshold

            # step 5 - update the precision recall curve
            target_classes = tc[hit].int()
            pred_classes_proba = pcp[best_pred][hit]
            if len(target_classes.shape) < len(pred_classes_proba.shape):
                target_classes = target_classes.unsqueeze(0)
            if hit.sum() > 0:
                self.average_precision.update(
                    pred_classes_proba, target_classes
                )
                self.hits += 1

    def compute(self):
        for pred, target in zip(self.pred_list, self.target_list):
            self.compute_image(pred, target)

        if self.hits > 0:
            return self.average_precision.compute()
        else:
            return torch.Tensor([torch.nan])

    def reset(self):
        self.pred_list = []
        self.target_list = []
        self.hits = 0
        self.average_precision.reset()

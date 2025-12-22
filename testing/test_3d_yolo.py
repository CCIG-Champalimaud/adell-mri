import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from adell_mri.modules.object_detection import YOLONet3d


def test_yolo():
    anchor_sizes = [[16, 16, 3], [32, 32, 5], [64, 64, 7]]
    yolo = YOLONet3d(
        "resnet", 1, n_classes=4, anchor_sizes=anchor_sizes, dev="cpu"
    )

    input_tensor = torch.ones([1, 1, 32, 32, 21])

    bb_center_pred, bb_size_pred, bb_object_pred, class_pred = yolo(
        input_tensor
    )

    logging.info("Input shape: %s", input_tensor.shape)
    logging.info("Center prediction shape: %s", bb_center_pred.shape)
    logging.info("Size prediction shape: %s", bb_size_pred.shape)
    logging.info("Objectness prediction shape: %s", bb_object_pred.shape)
    logging.info("Class prediction shape: %s", class_pred.shape)

    (
        bb_center_pred,
        bb_size_pred,
        bb_object_pred,
        class_pred,
    ) = yolo.channels_to_anchors(
        [bb_center_pred, bb_size_pred, bb_object_pred, class_pred]
    )

    logging.info("Testing prediction to bounding boxes")
    bb, scores, classification = yolo.recover_boxes(
        bb_center_pred[0], bb_size_pred[0], bb_object_pred[0], class_pred[0]
    )
    logging.info("Bounding box shape: %s", bb.shape)
    logging.info("Object scores shape: %s", scores.shape)

    assert len(bb.shape) == 2, "length of bb shape is wrong"
    assert len(scores.shape) == 1, "length of scores shape is wrong"

    logging.info("Testing prediction to bounding boxes with NMS")
    bb, scores, classification = yolo.recover_boxes(
        bb_center_pred[0],
        bb_size_pred[0],
        bb_object_pred[0],
        class_pred[0],
        nms=True,
    )
    logging.info("Bounding box shape: %s", bb.shape)
    logging.info("Object scores shape: %s", scores.shape)

    assert len(bb.shape) == 2, "length of bb shape is wrong for nms"
    assert len(scores.shape) == 1, "length of scores shape is wrong for nms"

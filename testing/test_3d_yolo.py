import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))

import torch

from lib.modules.object_detection import YOLONet3d

def test_yolo():
    anchor_sizes = [
        [16,16,3],
        [32,32,5],
        [64,64,7]
    ]
    yolo = YOLONet3d(1,n_c=4,anchor_sizes=anchor_sizes,dev='cpu')

    input_tensor = torch.ones([1,1,32,32,21])

    bb_center_pred,bb_size_pred,bb_object_pred,class_pred = yolo(input_tensor)

    print("Input shape:",input_tensor.shape)
    print("\tCenter prediction shape:",bb_center_pred.shape)
    print("\tSize prediction shape:",bb_size_pred.shape)
    print("\tObjectness prediction shape:",bb_object_pred.shape)
    print("\tClass prediction shape:",class_pred.shape)

    bb_center_pred,bb_size_pred,bb_object_pred,class_pred = yolo.channels_to_anchors(
        [bb_center_pred,bb_size_pred,bb_object_pred,class_pred])

    print("\tTesting prediction to bounding boxes")
    bb,scores,classification = yolo.recover_boxes(
        bb_center_pred[0],bb_size_pred[0],
        bb_object_pred[0],class_pred[0])
    print("\t\tBounding box shape:",bb.shape)
    print("\t\tObject scores shape:",scores.shape)

    assert len(bb.shape) == 2,"length of bb shape is wrong"
    assert len(scores.shape) == 1,"length of scores shape is wrong"

    print("\tTesting prediction to bounding boxes with NMS")
    bb,scores,classification = yolo.recover_boxes(
        bb_center_pred[0],bb_size_pred[0],
        bb_object_pred[0],class_pred[0],nms=True)
    print("\t\tBounding box shape:",bb.shape)
    print("\t\tObject scores shape:",scores.shape)

    assert len(bb.shape) == 2,"length of bb shape is wrong for nms"
    assert len(scores.shape) == 1,"length of scores shape is wrong for nms"

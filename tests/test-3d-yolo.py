import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import time
import torch
from torchsummary import summary

from lib.modules.object_detection import *
from lib.utils import *

anchor_sizes = [
    [16,16,3],
    [32,32,5],
    [64,64,7]
]
yolo = YOLONet3d(1,n_c=4,act_fn=torch.nn.PReLU,anchor_sizes=anchor_sizes,dev='cpu')

summary(yolo,(1,128,128,21),device='cpu')

input_tensor = torch.ones([1,1,128,128,21])

bb_center_pred,bb_size_pred,bb_object_pred,class_pred = yolo(input_tensor)

print("Input shape:",input_tensor.shape)
print("\tCenter prediction shape:",bb_center_pred.shape)
print("\tSize prediction shape:",bb_size_pred.shape)
print("\tObjectness prediction shape:",bb_object_pred.shape)
print("\tClass prediction shape:",class_pred.shape)

print("\tTesting prediction to bounding boxes")
a = time.time()
bb,scores = yolo.recover_boxes(bb_size_pred[0],bb_center_pred[0],bb_object_pred[0])
b = time.time()
print("\t\tBounding box shape:",bb.shape)
print("\t\tObject scores shape:",scores.shape)
print("\t\tTime ellapsed: {:.4f}s".format(b-a))

print("\tTesting prediction to bounding boxes with NMS")
a = time.time()
bb,scores = yolo.recover_boxes(bb_size_pred[0],bb_center_pred[0],bb_object_pred[0],True)
b = time.time()
print("\t\tBounding box shape:",bb.shape)
print("\t\tObject scores shape:",scores.shape)
print("\t\tTime ellapsed: {:.4f}s".format(b-a))

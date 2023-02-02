import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))

import torch

from lib.utils.batch_preprocessing import BatchPreprocessing,mixup,partial_mixup,label_smoothing

def test_label_smoothing():
    f = 0.2
    labels = torch.Tensor([0,1,1,0])
    expected_labels = torch.Tensor([0.2,0.8,0.8,0.2])
    labels = label_smoothing(labels,0.2)
    assert torch.all(expected_labels == labels)
    
def test_mixup():
    alpha = 0.2
    image = torch.ones([4,16,16])
    labels = torch.Tensor([0,1,1,0])
    mixup(image,labels,alpha)

def test_partial_mixup():
    alpha = 0.2
    image = torch.ones([4,16,16])
    labels = torch.Tensor([0,1,1,0])
    partial_mixup(image,labels,alpha)
    
def test_batch_preprocessing():
    f = 0.2
    alpha = 0.2
    p_m = 0.4
    bpp = BatchPreprocessing(f,alpha,p_m,42)
    image = torch.ones([4,16,16])
    labels = torch.Tensor([0,1,1,0])
    bpp(image,labels)
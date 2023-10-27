import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))

import torch
from lib.modules.semi_supervised_segmentation.losses import (
    AnatomicalContrastiveLoss, 
    PseudoLabelCrossEntropy,
    NearestNeighbourLoss)

bs,c,h,w,nc = 2,1,32,32,5
n_features = 128
trials = 1

def test_pseudo_label_cross_entropy_loss():
    plce = PseudoLabelCrossEntropy(threshold=0.5)

    pred = torch.randn([bs,nc,h,w])
    pseudo_labels = torch.rand([bs,nc,h,w])

    result = plce.forward(pred,pseudo_labels)
    assert len(result.shape) == 0
    assert torch.isnan(result) == False

def test_anatomical_contrastive_loss():
    for _ in range(trials):
        acl = AnatomicalContrastiveLoss(n_classes=nc,
                                        n_features=n_features,
                                        batch_size=bs,
                                        top_k=100)
        
        pred = torch.rand([bs,nc,h,w])
        features = torch.randn([bs,n_features,h,w])
        labels = (torch.rand([bs,nc,h,w]) > 0.5).float()

        result = acl.forward(pred, labels, features)

        assert len(result.shape) == 0
        assert torch.isnan(result) == False

def test_nearest_neighbour_loss():
    for _ in range(trials):
        nnl = NearestNeighbourLoss(100,
                                n_classes=nc,
                                max_elements_per_batch=100,
                                n_samples_per_class=10)
        
        def get_y_features():
            y = (torch.rand([bs,nc,h,w]) > 0.5).float()
            features = torch.rand([bs,n_features,h,w])
            return y, features

        for _ in range(4):
            y, features = get_y_features()
            nnl.put(features, y)

        y, features = get_y_features()
        result = nnl.forward(features, y)

        assert len(result.shape) == 0
        assert torch.isnan(result) == False

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from adell_mri.modules.layers.conv_next import ConvNeXt
from adell_mri.modules.classification import (
    HybridClassifier,
    TabularClassifier,
)

batch_size = 4
n_features = 10
c, h, w, d = [3, 64, 64, 32]
n_classes = 2


def test_tabular_classifier():
    tab_class = TabularClassifier(
        n_features, [64, 64], torch.nn.Identity, n_classes
    )
    input_tensor = torch.rand(batch_size, n_features)
    out_tensor = tab_class(input_tensor)
    assert list(out_tensor.shape) == [batch_size, n_classes - 1]


def test_hybrid_classifier():
    tab_class = TabularClassifier(
        n_features, [64, 64], torch.nn.Identity, n_classes
    )
    conv_class = ConvNeXt(
        {
            "spatial_dim": 2,
            "in_channels": c,
            "structure": [[32, 32, 3, 1], [32, 32, 3, 1]],
        },
        {"in_channels": 32, "structure": [64, 1]},
    )
    hybrid_class = HybridClassifier(
        tabular_module=tab_class, convolutional_module=conv_class
    )
    input_tensor_tab = torch.rand(batch_size, n_features)
    input_tensor_conv = torch.rand(batch_size, c, h, w)
    out_tensor = hybrid_class(input_tensor_conv, input_tensor_tab)
    assert list(out_tensor.shape) == [batch_size, n_classes - 1]

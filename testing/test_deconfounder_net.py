import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from lib.modules.classification.classification import (
    deconfounded_classification,
)

c, h, w, d = [1, 64, 64, 16]


def test_deconfounder_net():
    net = deconfounded_classification.DeconfoundedNet(
        n_features_deconfounder=128,
        n_cat_deconfounder=[2],
        n_cont_deconfounder=5,
    )
    input_tensor = torch.rand([2, c, h, w, d])
    out = net(input_tensor)

    classification, conf_class, reg_class, features = out

    assert list(classification.shape) == [2, 1]
    assert [list(x.shape) == [2, 2] for x in conf_class]
    assert list(reg_class.shape) == [2, 5]
    assert list(features.shape) == [2, 512]

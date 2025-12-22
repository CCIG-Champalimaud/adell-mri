import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from itertools import product

import pytest
import torch

from adell_mri.modules.classification.classification.multiple_instance_learning import (
    MultipleInstanceClassifier,
    get_adn_fn,
)

batch_size = 4
input_dim = 1
input_dim_mil = 32
classification_structure = [32, 16]
feat_extraction_structure = [32, 16]
n_slices = 8
n_classes = 2
adn_fn = get_adn_fn(1, "identity", "gelu", 0.1)


@pytest.mark.parametrize(
    "classification_mode,attention",
    product(["mean", "max", "vocabulary"], [True, False]),
)
def test_mil(classification_mode, attention):
    mod = MultipleInstanceClassifier(
        module=torch.nn.Conv2d(input_dim, input_dim_mil, 3),
        module_out_dim=input_dim_mil,
        n_classes=n_classes,
        feat_extraction_structure=feat_extraction_structure,
        classification_structure=classification_structure,
        classification_mode=classification_mode,
        classification_adn_fn=adn_fn,
        n_slices=n_slices,
        use_positional_embedding=False,
        attention=attention,
        dim=2,
    )
    input_tensor = torch.rand(batch_size, input_dim, 32, 32, n_slices)
    output = mod(input_tensor)
    assert list(output.shape) == [
        batch_size,
        1 if n_classes == 2 else n_classes,
    ]

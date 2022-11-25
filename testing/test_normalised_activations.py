import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import pytest

import torch
from lib.modules.activations import NormalizedActivation,activation_gradient_factory

act_list = list(activation_gradient_factory.keys())

@pytest.mark.parametrize("act_str",act_list)
def test_normalised_activations(act_str):
    norm_act = NormalizedActivation(act_str)
    input_tensor =  torch.rand(size=[1,16])
    output = norm_act(input_tensor)
    assert list(output.shape) == list(input_tensor.shape),\
        "output shape does not match input shape"
    assert torch.all(torch.isnan(output)) == False,\
        "operation produced nan"
    assert torch.all(torch.isinf(output)) == False,\
        "operation produced inf"
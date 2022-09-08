import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from lib.modules.activations import *

act_list = list(activation_gradient_factory.keys())

def test_normalised_activations():
    for act_str in act_list:
        norm_act = NormalizedActivation(act_str)
        input_tensor =  torch.rand(size=[1,16])
        output = norm_act(input_tensor)
        assert (list(output.shape) == list(input_tensor),
                "output shape does not match input shape")
        assert (torch.all(torch.isnan(output)) == False,
                "operation produced nan")
        assert (torch.all(torch.isinf(output)) == False,
                "operation produced inf")
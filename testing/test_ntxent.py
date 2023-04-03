import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))

import torch
from lib.modules.self_supervised import NTXent

def test_ntxent():
      X1,X2 = torch.rand(4,32),torch.rand(4,32)
      ntxent = NTXent()
      p = ntxent(X1,X2)


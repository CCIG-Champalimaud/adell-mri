import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from adell_mri.modules.object_detection import mAP


def test_yolo():
    map = mAP(3)

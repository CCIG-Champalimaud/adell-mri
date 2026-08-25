import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


from adell_mri.modules.object_detection import mAP


def test_yolo():
    map = mAP(3)

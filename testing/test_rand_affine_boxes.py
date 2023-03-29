import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))

import torch
import numpy as np
from lib.utils.monai_transforms import RandAffineWithBoxesd

c,h,w,d = 1,128,128,128
boxes = np.array([[0,1,2,4,2,5]]) * 4 + 32

def test_rand_affine_with_boxes():
    in_arr = np.zeros([c,h,w,d])
    for box in boxes:
        in_arr[:,box[0],box[1],box[2]] = 1
        in_arr[:,box[3],box[4],box[5]] = 1
        
    rawb = RandAffineWithBoxesd(image_keys=["image"],box_keys=["boxes"],
                                prob=1.0,
                                rotate_range=(1,1,1),
                                shear_range=(1,1,1),
                                translate_range=(1,1,1))
    
    output = rawb({"image":in_arr,"boxes":boxes})
    boxes_from_image = np.array(np.where(output["image"] > 0.1)[1:])
    tl = boxes_from_image.min(1)[np.newaxis,:]
    br = boxes_from_image.max(1)[np.newaxis,:]
    
    print(boxes)
    print(tl,br)
    print(rawb.last_affine)
    
test_rand_affine_with_boxes()
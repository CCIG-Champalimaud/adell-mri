import os
import argparse
import numpy as np
import monai
from skimage import measure
from glob import glob

def value_range(x):
    return x.min(),x.max()

def mask_to_bb(img:np.ndarray)->list:
    img = np.round(img)
    labelled_image = measure.label(img,connectivity=3)
    uniq = np.unique(labelled_image)
    uniq = uniq[uniq != 0]
    
    bb_vertices = []
    c = []
    for u in uniq:
        C = np.where(labelled_image == u)
        bb = np.array([value_range(c) for c in C])
        if np.all(bb[:,1] == bb[:,0]) == False:
            bb_vertices.append(bb)
            c.append(np.median(img[C]))

    return bb_vertices,c

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Returns the normalized coordinates for the bounding \
            boxes characterising each object.")

    parser.add_argument(
        '--input_path',dest="input_path",required=True,
        help="Path to folder containing nibabel compatible files")
    parser.add_argument(
        '--pattern',dest="pattern",default="*nii.gz",
        help="Pattern to match")
    parser.add_argument(
        '--split_character',dest="split_character",default=None,
        nargs='+',help="Splits the file name at these characters")
    
    args = parser.parse_args()

    t = monai.transforms.Compose([
        monai.transforms.LoadImaged(['image']),
        monai.transforms.AddChanneld(['image']),
        monai.transforms.Orientationd(['image'],"RAS")])

    for file_path in glob(os.path.join(args.input_path,args.pattern)):
        fdata = t({'image':file_path})['image'][0]
        sh = np.array(fdata.shape)
        image_id = file_path.split(os.sep)[-1]
        if args.split_character is not None:
            for char in args.split_character:
                image_id = image_id.split(char)[0]
        for bb,c in zip(*mask_to_bb(fdata)):
            x = image_id + ',' + ','.join([str(x) for x in bb.flatten("F")])
            x += ',' + ','.join([str(x) for x in sh])
            x += ',{}'.format(int(c))
            print(x)
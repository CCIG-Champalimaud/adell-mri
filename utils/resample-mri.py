import argparse
import numpy as np
import itk
import SimpleITK as sitk

def resample_image(sitk_image,out_spacing=[1.0, 1.0, 1.0],is_label=False):
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0.0)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    output = resample.Execute(sitk_image)
    for k in sitk_image.GetMetaDataKeys():
        v = sitk_image.GetMetaData(k)
        output.SetMetaData(k,v)

    return output

desc = """
Resamples an image to a target spacing.
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=desc)

    parser.add_argument(
        "--image_path",dest="image_path",required=True,
        type=str,help="Path to fixed image")
    parser.add_argument(
        '--spacing',dest='spacing',required=True,nargs='+',
        type=float,help="Target spacing")
    parser.add_argument(
        '--is_label',dest='is_label',action="store_true",
        default=False,help="Image is label map (uses NN interpolation)")
    parser.add_argument(
        "--output_path",dest="output_path",required=True,
        type=str,help="Path to output file (moved images)")

    args = parser.parse_args()

    # loading image

    image_path = args.image_path
    sf = sitk.sitkFloat32
    fixed_image = sitk.ReadImage(image_path,sf)

    # resampling to common space

    output_image = resample_image(fixed_image,args.spacing,args.is_label)
    sitk.WriteImage(output_image,args.output_path)

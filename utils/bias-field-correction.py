"""
Simple script to perform bias field correction. In essence it just uses 
SimpleITK to do all the heavy lifting.
"""

import argparse
import SimpleITK as sitk

def correct_bias_field(image,n_fitting_levels,n_iter,shrink_factor=1):
    image_ = image
    if shrink_factor > 1:
        image_ = sitk.Shrink(
            image_,[shrink_factor]*image_.GetDimension())
    #mask_image = sitk.OtsuThreshold(image_)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(n_fitting_levels*[n_iter])
    corrected_image = corrector.Execute(image_)
    log_bf = corrector.GetLogBiasFieldAsImage(image)
    corrected_input_image = image/sitk.Exp(log_bf)
    corrected_input_image = sitk.Cast(
        corrected_input_image,sitk.sitkFloat32)
    corrected_input_image.CopyInformation(image)
    for k in image.GetMetaDataKeys():
        v = image.GetMetaData(k)
        corrected_input_image.SetMetaData(k,v)
    return corrected_input_image

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Correct the bias field in MRI scans.')
    parser.add_argument(
        '--input_path',dest="input_path",type=str,
        help='path to image to be corrected')
    parser.add_argument(
        '--output_path',dest='output_path',type=str,
        help='path to output scan')
    parser.add_argument(
        '--n_fitting_levels',dest="n_fitting_levels",type=int,default=4,
        help="number of fitting levels (different bias frequencies)")
    parser.add_argument(
        '--n_iter',dest="n_iter",type=int,default=50,
        help="maximum number of iterations per fitting level")
    parser.add_argument(
        '--shrink_factor',dest="shrink_factor",type=int,default=1,
        help="shrink factor")
    args = parser.parse_args()

    input_image = sitk.ReadImage(args.input_path,sitk.sitkFloat32)
    corrected_image = correct_bias_field(
        input_image,args.n_fitting_levels,args.n_iter,args.shrink_factor)
    sitk.WriteImage(corrected_image,args.output_path)
"""
Simple script to perform bias field correction. In essence it just uses 
SimpleITK to do all the heavy lifting.
"""

import argparse
import SimpleITK as sitk

desc = "Correct the bias field in MRI scans."


def correct_bias_field(image, n_fitting_levels, n_iter, shrink_factor=1):
    image_ = image
    mask_image = sitk.OtsuThreshold(image_)
    if shrink_factor > 1:
        image_ = sitk.Shrink(image_, [shrink_factor] * image_.GetDimension())
        mask_image = sitk.Shrink(
            mask_image, [shrink_factor] * mask_image.GetDimension()
        )
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(n_fitting_levels * [n_iter])
    corrector.SetConvergenceThreshold(0.001)
    corrected_image = corrector.Execute(image_, mask_image)
    log_bf = corrector.GetLogBiasFieldAsImage(image)
    corrected_input_image = image / sitk.Exp(log_bf)
    corrected_input_image = sitk.Cast(corrected_input_image, sitk.sitkFloat32)
    corrected_input_image.CopyInformation(image)
    for k in image.GetMetaDataKeys():
        v = image.GetMetaData(k)
        corrected_input_image.SetMetaData(k, v)
    return corrected_input_image


def main(arguments):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--input_path",
        dest="input_path",
        type=str,
        help="path to image to be corrected",
    )
    parser.add_argument(
        "--output_path",
        dest="output_path",
        type=str,
        help="path to output scan",
    )
    parser.add_argument(
        "--n_fitting_levels",
        dest="n_fitting_levels",
        type=int,
        default=4,
        help="number of fitting levels (different bias frequencies)",
    )
    parser.add_argument(
        "--n_iter",
        dest="n_iter",
        type=int,
        default=50,
        help="maximum number of iterations per fitting level",
    )
    parser.add_argument(
        "--shrink_factor",
        dest="shrink_factor",
        type=int,
        default=1,
        help="shrink factor for bias field correction",
    )
    args = parser.parse_args(arguments)

    input_image = sitk.ReadImage(args.input_path, sitk.sitkFloat32)
    corrected_image = correct_bias_field(
        input_image, args.n_fitting_levels, args.n_iter, args.shrink_factor
    )
    sitk.WriteImage(corrected_image, args.output_path)

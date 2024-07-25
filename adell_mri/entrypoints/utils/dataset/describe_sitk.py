import argparse
import numpy as np
import SimpleITK as sitk

desc = "Describes SITK image properties."


def main(arguments):
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--sitk_paths",
        required=True,
        help="Paths to SITK images.",
        nargs="+",
    )

    args = parser.parse_args(arguments)

    for sitk_image_path in args.sitk_paths:
        image = sitk.ReadImage(sitk_image_path)
        image_array = sitk.GetArrayFromImage(image)
        u, c = np.unique(image_array, return_counts=True)
        print(f"Image: {sitk_image_path}")
        print(f"\tImage size: {image.GetSize()}")
        print(f"\tImage spacing: {image.GetSpacing()}")
        print(f"\tImage origin: {image.GetOrigin()}")
        print(f"\tImage direction: {image.GetDirection()}")
        print(
            f"\tImage number of components: {image.GetNumberOfComponentsPerPixel()}"
        )
        print(f"\tImage pixel type: {image.GetPixelIDTypeAsString()}")
        if u.shape[0] > 5:
            print(f"\tNumber of unique values: {len(u)}")
        else:
            print(f"\tUnique values: {[(i,j) for i,j in zip(u,c)]}")

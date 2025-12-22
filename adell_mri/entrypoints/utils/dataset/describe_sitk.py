desc = "Describes SITK image properties."


def main(arguments):
    import argparse

    import numpy as np
    import SimpleITK as sitk
    from adell_mri.utils.python_logging import get_logger

    logger = get_logger(__name__)

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--paths",
        required=True,
        help="Paths to SITK images.",
        nargs="+",
    )

    args = parser.parse_args(arguments)

    for sitk_image_path in args.paths:
        image = sitk.ReadImage(sitk_image_path)
        image_array = sitk.GetArrayFromImage(image)
        u, c = np.unique(image_array, return_counts=True)
        logger.info("Image: %s", sitk_image_path)
        logger.info("Image size: %s", image.GetSize())
        logger.info("Image spacing: %s", image.GetSpacing())
        logger.info("Image origin: %s", image.GetOrigin())
        logger.info("Image direction: %s", image.GetDirection())
        logger.info(
            "Image number of components: %s",
            image.GetNumberOfComponentsPerPixel(),
        )
        logger.info("Image pixel type: %s", image.GetPixelIDTypeAsString())
        if u.shape[0] > 5:
            logger.info("Number of unique values: %s", len(u))
        else:
            logger.info("Unique values: %s", [(i, j) for i, j in zip(u, c)])

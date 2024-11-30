import einops
import monai
import SimpleITK as sitk
import torch

from ...custom_types import TensorDict


def normalize_along_slice(
    X: torch.Tensor,
    min_value: float = 0.0,
    max_value: float = 1.0,
    dim: int = -1,
) -> torch.Tensor:
    """
    Performs minmax normalization along a given axis for a tensor.

    Args:
        X (torch.Tensor): tensor.
        min_value (float, optional): min value for output tensor. Defaults to
            0.0.
        max_value (float, optional): max value for output tensor. Defaults to
            1.0.
        dim (int, optional): dimension along which the minmax normalization is
            performed. Defaults to -1.

    Returns:
        torch.Tensor: minmax normalized tensor.
    """
    sh = X.shape
    assert dim < len(sh)
    assert (
        max_value > min_value
    ), "max_value {} must be larger than min_value {}".format(
        max_value, min_value
    )
    if dim < 0:
        dim = len(sh) + dim
    dims = ["c", "h", "w", "d"]
    lhs = " ".join(dims)
    rhs = "{} ({})".format(
        dims[dim], " ".join([d for d in dims if d != dims[dim]])
    )
    average_shape = [1 if i != dim else sh[dim] for i in range(len(sh))]
    flat_X = einops.rearrange(X, "{} -> {}".format(lhs, rhs))
    dim_max = flat_X.max(-1).values.reshape(average_shape)
    dim_min = flat_X.min(-1).values.reshape(average_shape)
    identical = dim_max == dim_min
    mult = torch.where(identical, 0.0, 1.0)
    denominator = torch.where(identical, 1.0, dim_max - dim_min)
    X = (X - dim_min) / denominator * mult
    X = X * (max_value - min_value) + min_value
    return X


class ConditionalRescaling(monai.transforms.Transform):
    """
    Rescales an array using scale if any value in the array is larger than
    max_value.
    """

    def __init__(self, max_value: float, scale: float):
        """
        Args:
            max_value (float): maximum value for condition.
            scale (float): scaling factor.
        """
        self.max_value = max_value
        self.scale = scale

    def __call__(self, X):
        if X.max() > self.max_value:
            X = X * self.scale
        return X


class ConditionalRescalingd(monai.transforms.Transform):
    """
    Dictionary version of ConditionalRescaling.
    """

    def __init__(self, keys: list[str], max_value: float, scale: float):
        """
        Args:
            keys (list[str]): keys to which conditional rescaling will be
                applied.
            max_value (float): maximum value for condition.
            scale (float): scaling factor.
        """
        self.keys = keys
        self.max_value = max_value
        self.scale = scale

        self.transforms = {
            k: ConditionalRescaling(self.max_value, self.scale)
            for k in self.keys
        }

    def __call__(self, data):
        for k in data:
            if k in self.transforms:
                data[k] = self.transforms[k](data[k])
        return data


class Offset(monai.transforms.Transform):
    """
    Offsets an array using an offset value (array = array - offset). If no
    value is provided, the minimum value for the array is used.
    """

    def __init__(self, offset: float = None):
        """
        Args:
            offset (float, optional): value for offset. Defaults to None.
        """
        self.offset = offset

    def __call__(self, data):
        offset = data.min() if self.offset is None else self.offset
        return data - offset


class Offsetd(monai.transforms.MapTransform):
    """
    Dictionary version of Offset.
    """

    def __init__(self, keys: list[str], offset: float = None):
        """
        Args:
            keys (list[str]): keys to offset.
            offset (float, optional): value for offset. Defaults to None.
        """

        self.keys = keys
        self.offset = offset
        self.tr = {k: Offset(offset) for k in self.keys}

    def __call__(self, data):
        for k in self.keys:
            data[k] = self.tr[k](data[k])
        return data


class BiasFieldCorrection(monai.transforms.Transform):
    """
    MONAI transform that automatically performs bias field correction using
    SITK.
    """

    def __init__(
        self, n_fitting_levels: int, n_iter: int, shrink_factor: float
    ):
        """
        Args:
            n_fitting_levels (int): number of fitting levels.
            n_iter (int): number of correction iterations.
            shrink_factor (float): shrink factor.
        """
        self.n_fitting_levels = n_fitting_levels
        self.n_iter = n_iter
        self.shrink_factor = shrink_factor

    def correct_bias_field(self, image):
        image_ = image
        mask_image = sitk.OtsuThreshold(image_)
        if self.shrink_factor > 1:
            image_ = sitk.Shrink(
                image_, [self.shrink_factor] * image_.GetDimension()
            )
            mask_image = sitk.Shrink(
                mask_image, [self.shrink_factor] * mask_image.GetDimension()
            )
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(
            self.n_fitting_levels * [self.n_iter]
        )
        corrector.SetConvergenceThreshold(0.001)
        corrector.Execute(image_, mask_image)
        log_bf = corrector.GetLogBiasFieldAsImage(image)
        corrected_input_image = image / sitk.Exp(log_bf)
        corrected_input_image = sitk.Cast(
            corrected_input_image, sitk.sitkFloat32
        )
        corrected_input_image.CopyInformation(image)
        for k in image.GetMetaDataKeys():
            v = image.GetMetaData(k)
            corrected_input_image.SetMetaData(k, v)
        return corrected_input_image

    def correct_bias_field_from_metadata_tensor(self, X):
        X_ = sitk.GetImageFromArray(X.data.numpy())
        X_ = self.correct_bias_field(X_)
        X_ = sitk.GetArrayFromImage(X_)
        X.data = X_
        return X_

    def __call__(self, X):
        return self.correct_bias_field_from_array(X)


class BiasFieldCorrectiond(monai.transforms.MapTransform):
    """
    Dictionary version of BiasFieldCorrection
    """

    def __init__(
        self,
        keys: list[str],
        n_fitting_levels: int,
        n_iter: int,
        shrink_factor: int,
    ):
        """
        Args:
            keys (list[str]): keys to apply bias field correction to.
            n_fitting_levels (int): number of fitting levels.
            n_iter (int): number of correction iterations.
            shrink_factor (float): shrink factor.
        """
        self.keys = keys
        self.n_fitting_levels = n_fitting_levels
        self.n_iter = n_iter
        self.shrink_factor = shrink_factor

        self.transform = BiasFieldCorrection(
            self.n_fitting_levels, self.n_iter, self.shrink_factor
        )

    def __call__(self, X):
        for k in self.keys:
            X[k] = self.transform(X[k])
        return X


class ScaleIntensityAlongDim(monai.transforms.Transform):
    """
    MONAI transform that applies normalize_along_slice to inputs. This
    normalizes individual slices along a given dimension dim.
    """

    def __init__(
        self, min_value: float = 0.0, max_value: float = 1.0, dim: int = -1
    ):
        """
        Args:
            min_value (float, optional): min value for output tensor. Defaults
                to 0.0.
            max_value (float, optional): max value for output tensor. Defaults
                to 1.0.
            dim (int, optional): dimension along which the minmax normalization
                is performed. Defaults to -1.
        """
        self.min_value = min_value
        self.max_value = max_value
        self.dim = dim

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return normalize_along_slice(
            X, min_value=self.min_value, max_value=self.max_value, dim=self.dim
        )


class ScaleIntensityAlongDimd(monai.transforms.MapTransform):
    """
    MONAI dict transform that applies normalize_along_slice to inputs. This
    normalizes individual slices along a given dimension dim.
    """

    def __init__(
        self,
        keys: list[str],
        min_value: float = 0.0,
        max_value: float = 1.0,
        dim: int = -1,
    ):
        """
        Args:
            min_value (float, optional): min value for output tensor. Defaults to
                0.0.
            max_value (float, optional): max value for output tensor. Defaults to
                0.0.
            dim (int, optional): dimension along which the minmax normalization is
                performed. Defaults to -1.
        """

        self.keys = keys
        self.min_value = min_value
        self.max_value = max_value
        self.dim = dim

        if isinstance(self.keys, str):
            self.keys = [self.keys]

        self.tr = ScaleIntensityAlongDim(
            min_value=min_value, max_value=max_value, dim=dim
        )

    def __call__(self, X: TensorDict) -> TensorDict:
        for k in self.keys:
            X[k] = self.tr(X[k])
        return X

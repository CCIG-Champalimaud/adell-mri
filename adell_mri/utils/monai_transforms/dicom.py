import monai
from pydicom import dcmread


class LoadIndividualDICOM(monai.transforms.Transform):
    """
    LoadIndividualDICOM reads a path to a DICOM file, converts it to an array
    using pydicom, and reshapes it to have a batch dimension.

    This is a thin wrapper around pydicom's dcmread function to load individual
    DICOM files and reshape them to NCHW format for batch processing.
    """

    def __init__(self):
        pass

    def __call__(self, X: str):
        out = dcmread(X).pixel_array
        if len(out.shape) == 2:
            out = out[None, :, :, None]
        elif len(out.shape) == 3:
            out = out[None, :, :, :]
        return out


class LoadIndividualDICOMd(monai.transforms.MapTransform):
    """LoadIndividualDICOMd applies LoadIndividualDICOM transform to multiple keys.

    LoadIndividualDICOMd takes a list of keys as input. For each key, it applies the
    LoadIndividualDICOM transform to the corresponding value in the input dictionary.

    Args:
        keys (List[str]): List of keys to apply transform to.

    """

    def __init__(self, keys: list[str]):
        self.keys = keys
        self.tr = LoadIndividualDICOM()

    def __call__(self, X: str):
        for k in self.keys:
            X[k] = self.tr(X[k])
        return X

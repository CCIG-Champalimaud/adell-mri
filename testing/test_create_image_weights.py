import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import monai

from adell_mri.utils import CreateImageAndWeightsd, ConditionalRescalingd

all_keys = ["image", "image_1", "image_2"]
non_adc_keys = ["image"]
adc_image_keys = ["image_1"]
intp_resampling_augmentations = ["bilinear" for _ in all_keys]
crop_size = [256, 256, 20]
rs = [
    monai.transforms.Spacingd(
        keys=all_keys,
        pixdim=[0.5, 0.5, 3.0],
        mode=intp_resampling_augmentations,
    )
]
adc_factor = 1 / 3
scaling_ops = []
scaling_ops.append(monai.transforms.ScaleIntensityd(non_adc_keys, 0, 1))
scaling_ops.append(ConditionalRescalingd(adc_image_keys, 1000, 0.001))
scaling_ops.append(
    monai.transforms.ScaleIntensityd(
        adc_image_keys, None, None, -(1 - adc_factor)
    )
)
crop_op = [
    monai.transforms.CenterSpatialCropd(all_keys, crop_size),
    monai.transforms.SpatialPadd(all_keys, crop_size),
]
t = [
    monai.transforms.LoadImaged(
        all_keys, ensure_channel_first=True, allow_missing_keys=True
    ),
    CreateImageAndWeightsd(["image", "image_1", "image_2"], [1] + crop_size),
    monai.transforms.Orientationd(all_keys, "RAS"),
    *rs,
    *scaling_ops,
    *crop_op,
    monai.transforms.EnsureTyped(all_keys),
]

t = monai.transforms.Compose(t)

d_complete = {
    "image": "/home/jose_almeida/data/ProCAncer-I/1/1/PCa-116121945215882851818757141286821804585/1.3.6.1.4.1.58108.1.192676062948176139650638721324387411054/image_T2.nii.gz",
    "image_1": "/home/jose_almeida/data/ProCAncer-I/1/1/PCa-116121945215882851818757141286821804585/1.3.6.1.4.1.58108.1.192676062948176139650638721324387411054/image_ADC.nii.gz",
    "image_2": "/home/jose_almeida/data/ProCAncer-I/1/1/PCa-116121945215882851818757141286821804585/1.3.6.1.4.1.58108.1.192676062948176139650638721324387411054/image_DWI.nii.gz",
}

d_missing = {
    "image_1": "/home/jose_almeida/data/ProCAncer-I/1/1/PCa-116121945215882851818757141286821804585/1.3.6.1.4.1.58108.1.192676062948176139650638721324387411054/image_ADC.nii.gz"
}


def test_complete():
    correct_weights = {"image": 1, "image_1": 1, "image_2": 1}
    out = t(d_complete)
    for k in ["image", "image_1", "image_2"]:
        assert k in out.keys()
        assert k + "_weight" in out.keys()
        assert list(out[k].shape) == [1] + crop_size
        assert out[k + "_weight"] == correct_weights[k]


def test_missing():
    correct_weights = {"image": 0, "image_1": 1, "image_2": 0}
    out = t(d_missing)
    for k in ["image", "image_1", "image_2"]:
        assert k in out.keys()
        assert k + "_weight" in out.keys()
        assert list(out[k].shape) == [1] + crop_size
        assert out[k + "_weight"] == correct_weights[k]

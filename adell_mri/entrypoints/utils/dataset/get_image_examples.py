desc = "Produces a image examples from a DICOM dataset upon transformations."


def main(arguments):
    import argparse
    import numpy as np
    import monai
    from pathlib import Path
    from skimage import io
    from tqdm import tqdm
    from adell_mri.transform_factory.transforms import ClassificationTransforms
    from adell_mri.utils.dataset import Dataset
    from adell_mri.entrypoints.assemble_args import Parser

    parser = Parser(description=desc)

    # params
    parser.add_argument_by_key(
        [
            "dataset_json",
            ["image_keys", "image_keys", {"required": True}],
            "target_spacing",
            "pad_size",
            "crop_size",
            "resize_size",
            "excluded_ids",
        ]
    )

    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args(arguments)

    transform_pipeline = ClassificationTransforms(
        keys=args.image_keys,
        adc_keys=[],
        clinical_feature_keys=[],
        target_spacing=args.target_spacing,
        crop_size=args.crop_size,
        pad_size=args.pad_size,
        image_masking=False,
        image_crop_from_mask=False,
        branched=False,
        target_size=args.resize_size,
    )

    transforms = transform_pipeline.transforms()

    dataset = Dataset(args.dataset_json)

    dataset.filter_dictionary(filters_presence=[*args.image_keys])
    dataset.subsample_dataset(excluded_key_list=args.excluded_ids)

    monai_dataset = monai.data.Dataset(dataset.to_datalist(), transforms)

    out_path = Path(args.output_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    for data in tqdm(monai_dataset):
        study_folder = out_path / data["identifier"]
        study_folder.mkdir(exist_ok=True, parents=True)
        for idx in range(data["image"].shape[0]):
            seq = data["image"][idx].numpy()
            seq = np.uint8((seq - seq.min()) / (seq.max() - seq.min()) * 255)
            if len(seq.shape) == 3:
                all_images_for_seq = [
                    seq[:, :, idx] for idx in range(seq.shape[2])
                ]
            else:
                all_images_for_seq = [seq]

            for i, img in enumerate(all_images_for_seq):
                img_path = study_folder / f"{idx}_{i}.png"
                io.imsave(img_path, img)

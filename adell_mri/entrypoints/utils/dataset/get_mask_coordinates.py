desc = "Produces a JSON file with all mask coordinates after applying spatial transforms."


def main(arguments):
    import argparse
    import json
    import monai
    import torch
    from tqdm import tqdm
    from adell_mri.transform_factory.transforms import ClassificationTransforms
    from adell_mri.utils.dataset import Dataset
    from adell_mri.entrypoints.assemble_args import Parser

    parser = argparse.ArgumentParser(description=desc)

    parser = Parser()

    # params
    parser.add_argument_by_key(
        [
            "dataset_json",
            ["image_keys", "image_keys", {"required": True}],
            ["mask_key", "mask_key", {"required": True}],
            "target_spacing",
            "pad_size",
            "crop_size",
            "resize_size",
            "excluded_ids",
        ]
    )

    parser.add_argument("--output_json", type=str, required=True)

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
        mask_key=args.mask_key,
        branched=False,
        target_size=args.resize_size,
    )

    transforms = transform_pipeline.transforms()

    dataset = Dataset(args.dataset_json)

    dataset.filter_dictionary(
        filters_presence=[*args.image_keys, args.mask_key]
    )
    dataset.subsample_dataset(excluded_key_list=args.excluded_ids)

    monai_dataset = monai.data.Dataset(dataset.to_datalist(), transforms)

    all_coords = {}
    for data in tqdm(monai_dataset):
        mask = data[args.mask_key][0]
        unique_vals = torch.unique(mask)
        unique_vals = unique_vals[unique_vals != 0]
        if len(unique_vals) > 0:
            all_coords[data["identifier"]] = {}
        for u in unique_vals:
            x, y, z = torch.where(mask == u)
            all_coords[data["identifier"]][int(u)] = {
                "x": x.numpy().tolist(),
                "y": y.numpy().tolist(),
                "z": z.numpy().tolist(),
            }

    with open(args.output_json, "w") as f:
        json.dump(all_coords, f, indent=2)

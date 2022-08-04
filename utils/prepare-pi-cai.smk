import os
import re
from glob import glob 

input_path = os.environ["PICAI_PATH"]
output_paths = {
    "resampled":os.environ["PICAI_PATH_RESAMPLED"],
    "corrected":os.environ["PICAI_PATH_RESAMPLED_CORRECTED"],
    "dataset_information":"dataset_information"}

for k in output_paths:
    os.makedirs(output_paths[k],exist_ok=True)

patterns = {
    "ADC":"*/*/*_adc.mha",
    "HBV":"*/*/*_hbv.mha",
    "T2W":"*/*/*_t2w.mha"}

output_size = []
output_spacing = []
output_resampled = []
output_corrected = []
for k in patterns:
    output_size.append(
        "{}/size.{}.PICAI".format(output_paths["dataset_information"],k))
    output_spacing.append(
        "{}/spacing.{}.PICAI".format(output_paths["dataset_information"],k))
    for path in glob(os.path.join(input_path,patterns[k])):
        o = path.replace(input_path,"").strip('/')
        out = os.path.join(output_paths["resampled"],o)
        output_resampled.append(out)
        out = os.path.join(output_paths["corrected"],o)
        output_corrected.append(out)
        sub_o = '_'.join(o.split(os.sep)[-1].split('_')[:2])

rule all:
    input:
        output_size,
        output_spacing,
        output_resampled,
        output_corrected

rule get_spacing:
    input:
        input_path
    output:
        os.path.join(output_paths["dataset_information"],"spacing.{mod}.PICAI")
    params:
        pattern=lambda wc: patterns[wc.mod],
        q=0.9,
        parameter="spacing"
    shell:
        """
        python3 utils/get-info.py \
    	    --input_dir {input_path} \
    	    --pattern {params.pattern} \
    	    --parameter {params.parameter} \
    	    --quantile {params.q} > {output}
        """

rule get_size:
    input:
        input_path
    output:
        os.path.join(output_paths["dataset_information"],"size.{mod}.PICAI")
    params:
        pattern=lambda wc: patterns[wc.mod],
        q=0.5,
        parameter="size"
    shell:
        """
        python3 utils/get-info.py \
    	    --input_dir {input_path} \
    	    --pattern {params.pattern} \
    	    --parameter {params.parameter} \
    	    --quantile {params.q} > {output}
        """

rule bias_correction:
    input:
        path=os.path.join(
            output_paths["resampled"],"{sub_dir}","{patient_id}","{patient_id}_{study_id}_{mod}.mha"),
    output:
        f=os.path.join(
            output_paths["corrected"],"{sub_dir}","{patient_id}","{patient_id}_{study_id}_{mod}.mha")
    params:
        mod=lambda wc: wc.mod.upper(),
        n_fitting_levels=3,
        n_iter=100,
        shrink_factor=2
    shell:
        """
        mkdir -p $(dirname {output.f})

        python3 utils/bias-field-correction.py \
            --input_path {input.path} \
            --output_path {output.f} \
            --n_fitting_levels {params.n_fitting_levels} \
            --n_iter {params.n_iter} \
            --shrink_factor {params.shrink_factor}
        """

rule resample:
    input:
        path=os.path.join(
            input_path,"{sub_dir}","{patient_id}","{patient_id}_{study_id}_{mod}.mha"),
        spacing=output_spacing
    output:
        f=os.path.join(
            output_paths["resampled"],"{sub_dir}","{patient_id}","{patient_id}_{study_id}_{mod}.mha")
    params:
        mod=lambda wc: wc.mod.upper(),
        di_path=output_paths["dataset_information"]
    shell:
        """
        mkdir -p $(dirname {output.f})$

        python3 utils/resample-mri.py \
            --image_path {input.path} \
            --spacing $(cat {params.di_path}/spacing.{params.mod}.PICAI | tr ',' ' ') \
            --output_path {output.f}
        """

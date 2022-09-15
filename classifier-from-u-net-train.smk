import os
import json
from sklearn.model_selection import KFold

json_file = "dataset_information/bb.pi-cai.clinical.json"
folds_file = "folds-ids.csv"
checkpoint_path = "models"
summary_path = "summaries"
metric_path = "metrics"
dataset_id = "picai"
project_name = "picai_classification"
label_key = "image_labels"
ckpt_patterns = {
    "regular": "models/unet-simsiam-3d/unet_simsiam.{combs}.lesion.3d.picai_fold{fold}_last.ckpt",
    "scratch": "models/unet-simsiam-3d/unet_simsiam.{combs}.lesion.3d.picai.scratch_fold{fold}_last.ckpt",
    "clinical": "models/unet-simsiam-3d/unet_simsiam.{combs}.lesion.3d.picai.clinical_fold{fold}_last.ckpt",
    "scratch_clinical": "models/unet-simsiam-3d/unet_simsiam.{combs}.lesion.3d.picai.scratch.clinical_fold{fold}_last.ckpt",
}

dataset_information = {
    "spacing":{
        "T2W":"dataset_information/spacing.T2W.PICAI",
        "DWI":"dataset_information/spacing.HBV.PICAI"},
    "size":{
        "T2W":"dataset_information/size.T2W.PICAI",
        "DWI":"dataset_information/size.HBV.PICAI"},
    "crop_size":{
        "T2W":"dataset_information/size.T2W.PICAI",
        "DWI":"dataset_information/size.HBV.PICAI"}}
size_div = {"T2W":2,"DWI":1}
for k in dataset_information:
    for kk in dataset_information[k]:
        di = open(dataset_information[k][kk]).read().strip().split(',')
        di = [float(x) for x in di]
        if k == "crop_size":
            di = [di[0]/size_div[kk],di[1]/size_div[kk],di[2]]
        dataset_information[k][kk] = di

# general training parameters
model_types = ["unet_simsiam",]
spatial_dims = ["3d"]
combinations = [
    ["image","image_1","image_2"]]
anatomies = ["lesion"]
comb_match = {
    "image":"T2W",
    "image_1":"ADC",
    "image_2":"DWI"
}
inv_comb_match = {
    "T2W":"image",
    "ADC":"image_1",
    "DWI":"image_2"}
all_clinical_keys = ["patient_age","psa","prostate_volume"]
possible_labels = [0,1,2,3,4,5]
positive_labels = [1,2,3,4,5]
loss_gamma = 1.0
max_epochs = 100
n_folds = 5
class_weights = {"gland":1,"lesion":5}
early_stopping = 20
adc_factor = 1/3
adc_image_keys = ["image_1"]
n_devices = 1
batch_size = 8
seed = 42

def get_combs(wc):
    x = [inv_comb_match[x] for x in wc.combs.split(':')]
    return x

def get_spacing(wc):
    C = get_combs(wc)
    if 'image' in C:
        return dataset_information["spacing"]["T2W"]
    else:
        return dataset_information["spacing"]["DWI"]

def get_size(wc):
    C = get_combs(wc)
    if 'image' in C:
        return dataset_information["size"]["T2W"]
    else:
        return dataset_information["size"]["DWI"]

def get_crop_size(wc):
    C = get_combs(wc)
    if 'image' in C:
        return dataset_information["crop_size"]["T2W"]
    else:
        return dataset_information["crop_size"]["DWI"]

def get_pp(wc):
    if wc.model_id == "unetpp":
        return "--unet_pp"
    else:
        return ""

def get_ckpts(wc):
    combs = wc.combs
    if wc.scratch_clinical == "": k = "regular"
    if wc.scratch_clinical == ".scratch": k = "scratch"
    if wc.scratch_clinical == ".clinical": k = "clinical"
    if wc.scratch_clinical == ".scratch.clinical": k = "scratch_clinical"
    return [
        ckpt_patterns[k].format(combs=combs,fold=i) 
        for i in range(n_folds)]

def get_clinical_features(wc):
    if "clinical" in wc.scratch_clinical:
        return "--feature_keys " + ' '.join(all_clinical_keys)
    else:
        return ""

metrics = []
for model_type in model_types:
    for spatial_dim in spatial_dims:
        output_folder = "{}/classifier.{}-{}".format(
            metric_path,model_type,spatial_dim)
        os.makedirs(output_folder,exist_ok=True)
        for combination in combinations:
            for anatomy in anatomies:
                comb_str = ':'.join(
                    [comb_match[x] for x in combination])
                output_metrics_path = "{}/{}.{}.{}.{}.csv".format(
                    output_folder,comb_str,anatomy,spatial_dim,dataset_id)
                output_metrics_scratch_path = "{}/{}.{}.{}.{}.scratch.csv".format(
                    output_folder,comb_str,anatomy,spatial_dim,dataset_id)
                output_metrics_clinical_path = "{}/{}.{}.{}.{}.clinical.csv".format(
                    output_folder,comb_str,anatomy,spatial_dim,dataset_id)
                output_metrics_scratch_clinical_path = "{}/{}.{}.{}.{}.scratch.clinical.csv".format(
                    output_folder,comb_str,anatomy,spatial_dim,dataset_id)

                metrics.append(output_metrics_path)
                metrics.append(output_metrics_scratch_path)
                metrics.append(output_metrics_clinical_path)
                metrics.append(output_metrics_scratch_clinical_path)

wildcard_constraints:
    anatomy="[a-zA-Z0-9]+",
    dataset_id="[a-zA-Z0-9]+",
    spatial_dim="(2d|3d)",
    model_id="(unet_simsiam)",
    scratch_clinical="(|\..*)"

rule all:
    input:
        metrics

rule train_models:
    input:
        json_file=json_file,
        config="config/u-net-{spatial_dim}.yaml",
        config_resnet="config/resnet-transfer.yaml",
        checkpoints=get_ckpts,
        folds="dataset_information/folds-ids.csv"
    output:
        metrics=os.path.join(
            metric_path,"classifier.{model_id}-{spatial_dim}",
            "{combs}.{anatomy}.{spatial_dim}.{dataset_id}{scratch_clinical}.csv")
    params:
        identifier="{model_id}_classifier.{combs}.{anatomy}.{spatial_dim}.{dataset_id}{scratch_clinical}",
        image_keys=get_combs,
        checkpoint_dir=os.path.join(
            checkpoint_path,"classifier.{model_id}-{spatial_dim}"),
        summary_dir=os.path.join(
            summary_path,"classifier.{model_id}-{spatial_dim}"),
        cw=lambda wc: class_weights[wc.anatomy],
        spacing=get_spacing,
        size=get_size,
        crop_size=get_crop_size,
        pp=get_pp,
        clinical_features=get_clinical_features
    shell:
        """
        python3 classifier-from-u-net-train.py \
            --dataset_json {input.json_file} \
            --image_keys {params.image_keys} \
            --label_key {label_key} \
            --target_spacing {params.spacing}  \
            --input_size {params.size} \
            --crop_size {params.crop_size} \
            --possible_labels {possible_labels} \
            --positive_labels {positive_labels} \
            --unet_config_file {input.config} \
            --dev cuda \
            --seed {seed} \
            --n_workers 8 \
            --max_epochs {max_epochs} \
            --class_weights {params.cw} \
            --swa \
            --checkpoint_dir {params.checkpoint_dir} \
            --checkpoint_name {params.identifier} \
            --summary_dir {params.summary_dir} \
            --summary_name {params.identifier} \
            --project_name {project_name} \
            --metric_path {output.metrics} \
            --augment \
            --batch_size {batch_size} \
            --early_stopping {early_stopping} \
            --adc_factor {adc_factor} \
            --adc_image_keys {adc_image_keys} \
            --n_devices {n_devices} \
            --res_config_file {input.config_resnet} \
            --unet_checkpoint {input.checkpoints} \
            --folds $(cat {input.folds} | tr '\n' ' ') \
            {params.pp} \
            {params.clinical_features}
        """
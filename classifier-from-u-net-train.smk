import os
import json
from sklearn.model_selection import KFold

json_file = "dataset_information/bb.pi-cai.json"
folds_file = "folds-ids.csv"
checkpoint_path = "models"
summary_path = "summaries"
metric_path = "metrics"
dataset_id = "picai"
project_name = "picai_classification"
label_key = "image_labels"
ckpt_pattern = "models/unet-simsiam-3d/unet_simsiam.{combs}.lesion.3d.picai.prior_fold{fold}_last.ckpt"
ckpt_scratch_pattern = "models/unet-simsiam-3d/unet_simsiam.{combs}.lesion.3d.picai.prior.scratch_fold{fold}_last.ckpt"
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
possible_labels = [0,1,2,3,4,5]
positive_labels = [1,2,3,4,5]
loss_gamma = 1.0
max_epochs = 100
n_folds = 5
class_weights = {"gland":1,"lesion":5}
early_stopping = 10
adc_factor = 1/3
adc_image_keys = ["image_1"]
n_devices = 1
batch_size = 16
seed = 42

# defining folds
folds = [
    x.strip().split(',') 
    for x in open(folds_file,'r').readlines()]
n_folds = len(folds)
ids_in_folds = []
for f in folds:
    ids_in_folds.extend(f)
all_ids = json.load(open(json_file)).keys()
all_other_ids = [x for x in all_ids if x not in ids_in_folds]
fold_idxs = KFold(n_folds,shuffle=True,random_state=seed).split(all_other_ids)
for i in range(n_folds):
    f = folds[i]
    f.extend([all_other_ids[idx] for idx in next(fold_idxs)[1]])
    folds[i] = ','.join(f)

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
    return [ckpt_pattern.format(combs=combs,fold=i) for i in range(n_folds)]

def get_ckpts_scratch(wc):
    combs = wc.combs
    return [ckpt_pattern.format(combs,i) for i in range(n_folds)]

metrics = []
for model_type in model_types:
    for spatial_dim in spatial_dims:
        output_folder = "{}/classifier-{}-{}".format(
            metric_path,model_type,spatial_dim)
        output_folder_scratch = "{}/classifier-{}-{}-scratch".format(
            metric_path,model_type,spatial_dim)
        os.makedirs(output_folder,exist_ok=True)
        for combination in combinations:
            for anatomy in anatomies:
                comb_str = ':'.join(
                    [comb_match[x] for x in combination])
                output_metrics_path = "{}/{}.{}.{}.{}.prior.csv".format(
                    output_folder,comb_str,anatomy,spatial_dim,dataset_id)
                output_metrics_scratch_path = "{}/{}.{}.{}.{}.prior.csv".format(
                    output_folder_scratch,comb_str,anatomy,spatial_dim,dataset_id)

                metrics.append(output_metrics_path)
                metrics.append(output_metrics_scratch_path)

wildcard_constraints:
    anatomy="[a-zA-Z0-9]+",
    dataset_id="[a-zA-Z0-9]+",
    spatial_dim="(2d|3d)",
    model_id="(unet_simsiam)"

rule all:
    input:
        metrics

rule train_models_prior:
    input:
        json_file=json_file,
        config="config/u-net-{spatial_dim}.yaml",
        config_resnet="config/resnet-transfer.yaml",
        checkpoints=get_ckpts
    output:
        metrics=os.path.join(
            metric_path,"classifier-{model_id}-{spatial_dim}",
            "{combs}.{anatomy}.{spatial_dim}.{dataset_id}.prior.csv")
    params:
        identifier="{model_id}_classifier.{combs}.{anatomy}.{spatial_dim}.{dataset_id}.prior",
        image_keys=get_combs,
        prior_key="gland",
        checkpoint_dir=os.path.join(
            checkpoint_path,"classifier-{model_id}-{spatial_dim}"),
        summary_dir=os.path.join(
            summary_path,"classifier-{model_id}-{spatial_dim}"),
        cw=lambda wc: class_weights[wc.anatomy],
        spacing=get_spacing,
        size=get_size,
        crop_size=get_crop_size,
        pp=get_pp
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
            --folds {folds} \
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
            --skip_mask_key {params.prior_key} \
            --unet_checkpoint {input.checkpoints} \
            {params.pp}
        """

rule train_models_scratch_prior:
    input:
        json_file=json_file,
        config="config/u-net-{spatial_dim}.yaml",
        config_resnet="config/resnet-transfer.yaml",
        checkpoints=get_ckpts
    output:
        metrics=os.path.join(
            metric_path,"classifier-{model_id}-{spatial_dim}-scratch",
            "{combs}.{anatomy}.{spatial_dim}.{dataset_id}.prior.csv")
    params:
        identifier="{model_id}_classifier.{combs}.{anatomy}.{spatial_dim}.{dataset_id}.prior.scratch",
        image_keys=get_combs,
        prior_key="gland",
        checkpoint_dir=os.path.join(
            checkpoint_path,"classifier-{model_id}-{spatial_dim}"),
        summary_dir=os.path.join(
            summary_path,"classifier-{model_id}-{spatial_dim}"),
        cw=lambda wc: class_weights[wc.anatomy],
        spacing=get_spacing,
        size=get_size,
        crop_size=get_crop_size,
        pp=get_pp
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
            --folds {folds} \
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
            --skip_mask_key {params.prior_key} \
            --unet_checkpoint {input.checkpoints} \
            {params.pp}
        """

import os

json_file="dataset_information/bb.pi-cai.nc.json"
checkpoint_path = "models"
summary_path = "summaries"
metric_path = "metrics"
dataset_id = "picai"
ssl_model_id = "simsiam"
project_name = "picai_segmentation"
ssl_checkpoint_base = "models/{ssl_model_id}/{ssl_model_id}.{combs}.picai_fold{fold}_last.ckpt"
best_folds_ssl = {
    "T2W":0,"ADC":0,"DWI":0,
    "ADC:DWI":0,"T2W:ADC:DWI":0}
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
s = 0
for k in dataset_information:
    for kk in dataset_information[k]:
        di = open(dataset_information[k][kk]).read().strip().split(',')
        di = [float(x) for x in di]
        if k == "crop_size":
            di = [di[0]/size_div[kk]+s,di[1]/size_div[kk]+s,di[2]]
        dataset_information[k][kk] = di

model_types = ["unet",
               #"unetpp"
               ]
spatial_dims = ["3d"]
combinations = [
    #["image"],["image_1"],["image_2"],
    ["image_1","image_2"],
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
class_weights = {"gland":10,"lesion":250}
early_stopping = 15
adc_factor = 1/3
adc_image_keys = ["image_1"]
n_devices = 2

def get_combs(wc):
    x = [inv_comb_match[x] for x in wc.combs.split(':')]
    return x

def get_ssl_checkpoint(wc):
    C = wc.combs
    return ssl_checkpoint_base.format(
        ssl_model_id=wc.ssl_model_id,combs=C,fold=best_folds_ssl[C])

def get_masks(wc):
    C = get_combs(wc)
    masks = []
    masks.append("lesion_merge")
    return masks

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

metrics = []
for model_type in model_types:
    for spatial_dim in spatial_dims:
        output_folder = "{}/{}-{}-{}".format(
            metric_path,model_type,spatial_dim,ssl_model_id)
        output_folder_scratch = "{}/{}-{}-{}-scratch".format(
            metric_path,model_type,spatial_dim,ssl_model_id)
        os.makedirs(output_folder,exist_ok=True)
        for combination in combinations:
            for anatomy in anatomies:
                comb_str = ':'.join(
                    [comb_match[x] for x in combination])
                output_metrics_path = "{}/{}.{}.{}.{}.csv".format(
                    output_folder,comb_str,anatomy,spatial_dim,dataset_id)
                output_metrics_prior_path = "{}/{}.{}.{}.{}.prior.csv".format(
                    output_folder,comb_str,anatomy,spatial_dim,dataset_id)
                output_metrics_scratch_path = "{}/{}.{}.{}.{}.csv".format(
                    output_folder_scratch,comb_str,anatomy,spatial_dim,dataset_id)
                output_metrics_scratch_prior_path = "{}/{}.{}.{}.{}.prior.csv".format(
                    output_folder_scratch,comb_str,anatomy,spatial_dim,dataset_id)

                metrics.append(output_metrics_path)
                metrics.append(output_metrics_scratch_path)
                if anatomy == "lesion":
                    metrics.append(output_metrics_prior_path)
                    metrics.append(output_metrics_scratch_prior_path)

wildcard_constraints:
    anatomy="[a-zA-Z0-9]+",
    dataset_id="[a-zA-Z0-9]+",
    spatial_dim="(2d|3d)",
    model_id="(unet|unetpp)"

rule all:
    input:
        metrics

rule train_models:
    input:
        json_file=json_file,
        config="config/u-net-{spatial_dim}.yaml",
        config_resnet="config/resnet-transfer.yaml",
        ssl_checkpoint=get_ssl_checkpoint
    output:
        metrics=os.path.join(
            metric_path,"{model_id}-{spatial_dim}-{ssl_model_id}",
            "{combs}.{anatomy}.{spatial_dim}.{dataset_id}.csv")
    params:
        identifier="{model_id}_{ssl_model_id}.{combs}.{anatomy}.{spatial_dim}.{dataset_id}",
        image_keys=get_combs,
        mask_keys=get_masks,
        checkpoint_dir=os.path.join(
            checkpoint_path,"{model_id}-{ssl_model_id}-{spatial_dim}"),
        summary_dir=os.path.join(
            summary_path,"{model_id}-{ssl_model_id}-{spatial_dim}"),
        cw=lambda wc: class_weights[wc.anatomy],
        spacing=get_spacing,
        size=get_size,
        crop_size=get_crop_size,
        pp=get_pp
    shell:
        """
        python3 u-net-train.py \
            --dataset_json {input.json_file} \
            --image_keys {params.image_keys} \
            --mask_keys {params.mask_keys} \
            --target_spacing {params.spacing}  \
            --crop_size {params.crop_size} \
            --possible_labels {possible_labels} \
            --positive_labels {positive_labels} \
            --config_file {input.config} \
            --dev cuda \
            --seed 42 \
            --n_workers 8 \
            --loss_gamma {loss_gamma} \
            --loss_comb 0.5 \
            --max_epochs {max_epochs} \
            --n_folds {n_folds} \
            --class_weights {params.cw} \
            --swa \
            --checkpoint_dir {params.checkpoint_dir} \
            --checkpoint_name {params.identifier} \
            --summary_dir {params.summary_dir} \
            --summary_name {params.identifier} \
            --project_name {project_name} \
            --metric_path {output.metrics} \
            --augment \
            --early_stopping {early_stopping} \
            --adc_factor {adc_factor} \
            --adc_image_keys {adc_image_keys} \
            --n_devices {n_devices} \
            --res_config_file {input.config_resnet} \
            --res_checkpoint {input.ssl_checkpoint} \
            {params.pp} 
        """

rule train_models_scratch:
    input:
        json_file=json_file,
        config="config/u-net-{spatial_dim}.yaml",
        config_resnet="config/resnet-transfer.yaml"
    output:
        metrics=os.path.join(
            metric_path,"{model_id}-{spatial_dim}-{ssl_model_id}-scratch",
            "{combs}.{anatomy}.{spatial_dim}.{dataset_id}.csv")
    params:
        identifier="{model_id}_{ssl_model_id}.{combs}.{anatomy}.{spatial_dim}.{dataset_id}.scratch",
        image_keys=get_combs,
        mask_keys=get_masks,
        checkpoint_dir=os.path.join(
            checkpoint_path,"{model_id}-{ssl_model_id}-{spatial_dim}"),
        summary_dir=os.path.join(
            summary_path,"{model_id}-{ssl_model_id}-{spatial_dim}"),
        cw=lambda wc: class_weights[wc.anatomy],
        spacing=get_spacing,
        size=get_size,
        crop_size=get_crop_size,
        pp=get_pp
    shell:
        """
        python3 u-net-train.py \
            --dataset_json {input.json_file} \
            --image_keys {params.image_keys} \
            --mask_keys {params.mask_keys} \
            --target_spacing {params.spacing}  \
            --crop_size {params.crop_size} \
            --possible_labels {possible_labels} \
            --positive_labels {positive_labels} \
            --config_file {input.config} \
            --dev cuda \
            --seed 42 \
            --n_workers 8 \
            --loss_gamma {loss_gamma} \
            --loss_comb 0.5 \
            --max_epochs {max_epochs} \
            --n_folds {n_folds} \
            --class_weights {params.cw} \
            --swa \
            --checkpoint_dir {params.checkpoint_dir} \
            --checkpoint_name {params.identifier} \
            --summary_dir {params.summary_dir} \
            --summary_name {params.identifier} \
            --project_name {project_name} \
            --metric_path {output.metrics} \
            --augment \
            --early_stopping {early_stopping} \
            --adc_factor {adc_factor} \
            --adc_image_keys {adc_image_keys} \
            --n_devices {n_devices} \
            --res_config_file {input.config_resnet} \
            {params.pp} 
        """

rule train_models_prior:
    input:
        json_file=json_file,
        config="config/u-net-{spatial_dim}.yaml",
        config_resnet="config/resnet-transfer.yaml",
        ssl_checkpoint=get_ssl_checkpoint
    output:
        metrics=os.path.join(
            metric_path,"{model_id}-{spatial_dim}-{ssl_model_id}",
            "{combs}.{anatomy}.{spatial_dim}.{dataset_id}.prior.csv")
    params:
        identifier="{model_id}_{ssl_model_id}.{combs}.{anatomy}.{spatial_dim}.{dataset_id}.prior",
        image_keys=get_combs,
        prior_key="gland",
        mask_keys=get_masks,
        checkpoint_dir=os.path.join(
            checkpoint_path,"{model_id}-{ssl_model_id}-{spatial_dim}"),
        summary_dir=os.path.join(
            summary_path,"{model_id}-{ssl_model_id}-{spatial_dim}"),
        config="config/u-net-{spatial_dim}.yaml",
        cw=lambda wc: class_weights[wc.anatomy],
        spacing=get_spacing,
        size=get_size,
        crop_size=get_crop_size,
        pp=get_pp,
        ssl_checkpoint=get_ssl_checkpoint
    shell:
        """
        python3 u-net-train.py \
            --dataset_json {input.json_file} \
            --image_keys {params.image_keys} \
            --mask_keys {params.mask_keys} \
            --target_spacing {params.spacing}  \
            --crop_size {params.crop_size} \
            --possible_labels {possible_labels} \
            --positive_labels {positive_labels} \
            --config_file {input.config} \
            --dev cuda \
            --seed 42 \
            --n_workers 8 \
            --loss_gamma {loss_gamma} \
            --loss_comb 0.5 \
            --max_epochs {max_epochs} \
            --n_folds {n_folds} \
            --class_weights {params.cw} \
            --swa \
            --checkpoint_dir {params.checkpoint_dir} \
            --checkpoint_name {params.identifier} \
            --summary_dir {params.summary_dir} \
            --summary_name {params.identifier} \
            --project_name {project_name} \
            --metric_path {output.metrics} \
            --augment \
            --early_stopping {early_stopping} \
            --adc_factor {adc_factor} \
            --adc_image_keys {adc_image_keys} \
            --n_devices {n_devices} \
            --res_config_file {input.config_resnet} \
            --res_checkpoint {input.ssl_checkpoint} \
            --skip_mask_key {params.prior_key} \
            --input_size {params.size} \
            --resize_keys {params.prior_key} \
            {params.pp}
        """

rule train_models_scratch_prior:
    input:
        json_file=json_file,
        config="config/u-net-{spatial_dim}.yaml",
        config_resnet="config/resnet-transfer.yaml"
    output:
        metrics=os.path.join(
            metric_path,"{model_id}-{spatial_dim}-{ssl_model_id}-scratch",
            "{combs}.{anatomy}.{spatial_dim}.{dataset_id}.prior.csv")
    params:
        identifier="{model_id}_{ssl_model_id}.{combs}.{anatomy}.{spatial_dim}.{dataset_id}.prior.scratch",
        image_keys=get_combs,
        prior_key="gland",
        mask_keys=get_masks,
        checkpoint_dir=os.path.join(
            checkpoint_path,"{model_id}-{ssl_model_id}-{spatial_dim}"),
        summary_dir=os.path.join(
            summary_path,"{model_id}-{ssl_model_id}-{spatial_dim}"),
        config="config/u-net-{spatial_dim}.yaml",
        cw=lambda wc: class_weights[wc.anatomy],
        spacing=get_spacing,
        size=get_size,
        crop_size=get_crop_size,
        pp=get_pp
    shell:
        """
        python3 u-net-train.py \
            --dataset_json {input.json_file} \
            --image_keys {params.image_keys} \
            --mask_keys {params.mask_keys} \
            --target_spacing {params.spacing}  \
            --crop_size {params.crop_size} \
            --possible_labels {possible_labels} \
            --positive_labels {positive_labels} \
            --config_file {input.config} \
            --dev cuda \
            --seed 42 \
            --n_workers 8 \
            --loss_gamma {loss_gamma} \
            --loss_comb 0.5 \
            --max_epochs {max_epochs} \
            --n_folds {n_folds} \
            --class_weights {params.cw} \
            --swa \
            --checkpoint_dir {params.checkpoint_dir} \
            --checkpoint_name {params.identifier} \
            --summary_dir {params.summary_dir} \
            --summary_name {params.identifier} \
            --project_name {project_name} \
            --metric_path {output.metrics} \
            --augment \
            --early_stopping {early_stopping} \
            --adc_factor {adc_factor} \
            --adc_image_keys {adc_image_keys} \
            --n_devices {n_devices} \
            --res_config_file {input.config_resnet} \
            --skip_mask_key {params.prior_key} \
            --input_size {params.size} \
            --resize_keys {params.prior_key} \
            {params.pp}
        """

import os

json_file="dataset_information/bb.pi-cai.json"
checkpoint_path = "models"
summary_path = "summaries"
metric_path = "metrics"
dataset_id = "picai"
project_name = "picai_ssl"
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
            di = [round(di[0]/size_div[kk]),
                  round(di[1]/size_div[kk]),
                  round(di[2])]
        if k == "size":
            di = [round(di[0]),round(di[1]),round(di[2])]
        dataset_information[k][kk] = di

model_types = [
    "simsiam",
    #"byol"
    ]
spatial_dims = ["3d"]
combinations = [
    ["image"],["image_1"],["image_2"],
    ["image","image_1","image_2"],
    ["image","image_1"],
    ["image_1","image_2"]
    ]
comb_match = {
    "image":"T2W",
    "image_1":"ADC",
    "image_2":"DWI"
}
inv_comb_match = {
    "T2W":"image",
    "ADC":"image_1",
    "DWI":"image_2"}
max_epochs = 100
n_folds = 1
adc_factor = 1/3
adc_image_keys = ["image_1"]
n_devices = 2

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

def get_ema(wc):
    if wc.model_id == "byol":
        return "--ema"
    else:
        return ""

metrics = []
for model_type in model_types:
    for spatial_dim in spatial_dims:
        output_folder = "{}/{}".format(
            metric_path,model_type)
        os.makedirs(output_folder,exist_ok=True)
        for combination in combinations:
            comb_str = ':'.join(
                [comb_match[x] for x in combination])
            output_metrics_path = "{}/{}.{}.csv".format(
                output_folder,comb_str,dataset_id)
            output_metrics_prior_path = "{}/{}.{}.prior.csv".format(
                output_folder,comb_str,dataset_id)
            metrics.append(output_metrics_path)

wildcard_constraints:
    dataset_id="[a-zA-Z0-9]+",
    model_id="(simsiam|byol)"

rule all:
    input:
        metrics

rule train_models:
    input:
        json_file=json_file,
        config="config/simsiam.yaml",
    output:
        metrics=os.path.join(
            metric_path,"{model_id}",
            "{combs}.{dataset_id}.csv")
    params:
        identifier="{model_id}.{combs}.{dataset_id}",
        image_keys=get_combs,
        checkpoint_dir=os.path.join(
            checkpoint_path,"{model_id}"),
        summary_dir=os.path.join(
            summary_path,"{model_id}"),
        spacing=get_spacing,
        size=get_size,
        crop_size=get_crop_size,
        ema=get_ema
    shell:
        """
        python3 self-supervised-train.py \
            --dataset_json dataset_information/bb.pi-cai.json \
            --crop_size {params.crop_size} \
            --target_spacing {params.spacing} \
            --image_keys {params.image_keys} \
            --adc_image_keys image_1 \
            --adc_factor 0.3333333 \
            --config_file {input.config} \
            --dev "cuda" \
            --seed 42 \
            --n_workers 8 \
            --n_devices {n_devices} \
            --pre_load \
            --n_folds {n_folds} \
            --dropout_param 0.0 \
            --metric_path {output.metrics} \
            --summary_name {params.identifier} \
            --summary_dir {params.summary_dir} \
            --checkpoint_name {params.identifier} \
            --checkpoint_dir {params.checkpoint_dir} \
            --max_epochs {max_epochs} \
            --project_name {project_name} \
            {params.ema}
        """
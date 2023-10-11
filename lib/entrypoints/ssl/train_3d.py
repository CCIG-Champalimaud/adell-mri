import random
import json
import numpy as np
import torch
import monai
from copy import deepcopy

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar

from ...entrypoints.assemble_args import Parser
from ...utils import (
    collate_last_slice,
    RandomSlices,
    safe_collate)
from ...utils.pl_utils import get_ckpt_callback,get_logger,get_devices
from ...modules.self_supervised.pl import (
    SelfSLResNetPL,SelfSLUNetPL)
from ...utils import ExponentialMovingAverage
from ...modules.config_parsing import parse_config_ssl,parse_config_unet
from ...monai_transforms import (
    get_pre_transforms_ssl,get_post_transforms_ssl,get_augmentations_ssl)

torch.backends.cudnn.benchmark = True

def keep_first_not_none(*args):
    for arg in args:
        if arg is not None:
            return arg

def force_cudnn_initialization():
    """Convenience function to initialise CuDNN (and avoid the lazy loading
    from PyTorch).
    """
    s = 16
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s,s,s,s,device=dev), 
                               torch.zeros(s,s,s,s,device=dev))

def main(arguments):
    parser = Parser()

    parser.add_argument_by_key([
        "dataset_json",
        "image_keys",
        ("adc_keys","adc_image_keys"),
        "train_pids",
        "target_spacing",
        "pad_size","crop_size","random_crop_size",
        "different_crop",
        "subsample_size",
        "cache_rate",
        "precision",
        "unet_encoder",
        "batch_size",
        "config_file","ssl_method","ema",
        "checkpoint_dir","checkpoint_name","checkpoint","resume_from_last",
        "project_name","resume","summary_name","summary_dir","metric_path",
        "dev","n_workers",
        "seed",
        "max_epochs",
        "accumulate_grad_batches","gradient_clip_val",
        "dropout_param"
    ])
    
    args = parser.parse_args(arguments)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    output_file = open(args.metric_path,'w')

    keys = args.image_keys
    copied_keys = [k+"_copy" for k in keys]
    key_correspondence = {k:kk for k,kk in zip(keys,copied_keys)}
    if args.adc_image_keys is None:
        args.adc_image_keys = []
    args.adc_image_keys = [k for k in args.adc_image_keys if k in keys]
    intp = []
    intp_resampling_augmentations = []
    for k in keys:
        intp.append("area")
        intp_resampling_augmentations.append("bilinear")
    non_adc_keys = [k for k in keys if k not in args.adc_image_keys]
    all_keys = [*keys]

    data_dict = json.load(open(args.dataset_json,'r'))
    data_dict = {
        k:data_dict[k] for k in data_dict
        if len(set.intersection(set(data_dict[k]),
                                set(all_keys))) == len(all_keys)}
    if args.subsample_size is not None:
        ss = np.random.choice(
            list(data_dict.keys()),args.subsample_size,replace=False)
        data_dict = {k:data_dict[k] for k in ss}
    for k in data_dict:
        data_dict[k]["pid"] = k

    if args.unet_encoder is True:
        network_config,_ = parse_config_unet(
            args.config_file,len(keys),2)
        network_config_correct = deepcopy(network_config)
        for k in network_config:
            if k in ["loss_fn"]:
                del network_config_correct[k]
        n_dims = network_config["spatial_dimensions"]
    else:
        network_config,network_config_correct = parse_config_ssl(
            args.config_file,args.dropout_param,len(keys))
        n_dims = network_config["backbone_args"]["spatial_dim"]

    if (args.batch_size is not None) and (args.batch_size != "tune"):
        network_config["batch_size"] = args.batch_size
        network_config_correct["batch_size"] = args.batch_size

    if args.ema is True:
        bs = network_config_correct["batch_size"]
        ema_params = {
            "decay":0.99,
            "final_decay":1.0,
            "n_steps":args.max_epochs*len(data_dict)/bs}
        ema = ExponentialMovingAverage(**ema_params)
    else:
        ema = None

    if args.random_crop_size is None:
        roi_size = [64,64,8]
    else:
        roi_size = [int(x) for x in args.random_crop_size]

    is_ijepa = args.ssl_method is "ijepa"
    pre_transform_args = {
        "all_keys":all_keys,
        "copied_keys":copied_keys,
        "adc_keys":[],
        "non_adc_keys":non_adc_keys,
        "target_spacing":args.target_spacing,
        "crop_size":args.crop_size,
        "pad_size":args.pad_size,
        "n_channels":1,
        "n_dim":3,
        "skip_augmentations":is_ijepa,
        "jpeg_dataset":False}
    
    post_transform_args = {
        "all_keys":all_keys,
        "copied_keys":copied_keys,
        "skip_augmentations":is_ijepa}
    
    augmentation_args = {
        "all_keys":all_keys,
        "copied_keys":copied_keys if is_ijepa is False else [],
        "scaled_crop_size":args.scaled_crop_size,
        "roi_size":roi_size,
        "different_crop":args.different_crop,
        "vicregl":args.ssl_method == "vicregl",
        "n_transforms":args.n_transforms,
        "n_dim":3,
        "skip_augmentations":is_ijepa}
    
    if is_ijepa is True:
        image_size = keep_first_not_none(
            args.scaled_crop_size,args.crop_size)
        patch_size = network_config_correct["backbone_args"]["patch_size"]
        feature_map_size = [i//pi for i,pi in zip(image_size,patch_size)]
        network_config_correct["backbone_args"]["image_size"] = image_size
        network_config_correct["feature_map_dimensions"] = feature_map_size

    transforms = [
        *get_pre_transforms_ssl(**pre_transform_args),
        *get_augmentations_ssl(**augmentation_args),
        *get_post_transforms_ssl(**post_transform_args)]

    if n_dims == 2:
        transforms.append(
            RandomSlices(["image"],None,8,base=0.001))
        collate_fn = collate_last_slice
    else:
        collate_fn = safe_collate

    if args.train_pids is not None:
        train_pids = args.train_pids
    else:
        train_pids = [pid for pid in data_dict]
    train_list = [data_dict[pid] for pid in data_dict
                  if pid in train_pids]

    accelerator,devices,strategy = get_devices(args.dev)

    # split workers across cache construction and data loading
    a = args.n_workers // 2
    b = args.n_workers - a
    train_dataset = monai.data.CacheDataset(
        train_list,
        monai.transforms.Compose(transforms),
        cache_rate=args.cache_rate,num_workers=a)

    def train_loader_call(batch_size,shuffle=True): 
        return monai.data.ThreadDataLoader(
            train_dataset,batch_size=batch_size,
            shuffle=shuffle,num_workers=b,generator=g,
            collate_fn=collate_fn,pin_memory=True,
            persistent_workers=True,drop_last=True)

    train_loader = train_loader_call(
        network_config_correct["batch_size"],False)

    if args.unet_encoder is True:
        ssl = SelfSLUNetPL(
            training_dataloader_call=train_loader_call,
            aug_image_key_1="augmented_image_1",
            aug_image_key_2="augmented_image_2",
            box_key_1="box_1",
            box_key_2="box_2",
            n_epochs=args.max_epochs,
            vic_reg=args.ssl_method == "vicreg",
            vic_reg_local=args.ssl_method == "vicregl",
            ema=ema,
            **network_config_correct)
    else:
        ssl = SelfSLResNetPL(
            training_dataloader_call=train_loader_call,
            aug_image_key_1="augmented_image_1",
            aug_image_key_2="augmented_image_2",
            box_key_1="box_1",
            box_key_2="box_2",
            n_epochs=args.max_epochs,
            vic_reg=args.ssl_method == "vicreg",
            vic_reg_local=args.ssl_method == "vicregl",
            ema=ema,
            **network_config_correct)
    if "cuda" in args.dev:
        ssl = ssl.to("cuda")

    if args.checkpoint is not None:
        state_dict = torch.load(
            args.checkpoint,map_location=args.dev)['state_dict']
        inc = ssl.load_state_dict(state_dict)
    
    callbacks = [RichProgressBar()]

    ckpt_callback,ckpt_path,status = get_ckpt_callback(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        max_epochs=args.max_epochs,
        resume_from_last=args.resume_from_last,
        val_fold=None,
        monitor="val_loss")
    if ckpt_callback is not None:   
        callbacks.append(ckpt_callback)
    ckpt = ckpt_callback is not None
    if status == "finished":
        print("Training has finished")
        exit()
    
    logger = get_logger(summary_name=args.summary_name,
                        summary_dir=args.summary_dir,
                        project_name=args.project_name,
                        resume=args.resume,
                        fold=None)

    precision = {"16":16,"32":32,"bf16":"bf16"}[args.precision]
    trainer = Trainer(
        accelerator="gpu" if "cuda" in args.dev else "cpu",
        devices=devices,logger=logger,callbacks=callbacks,
        strategy=strategy,max_epochs=args.max_epochs,
        sync_batchnorm=True if strategy is not None else False,
        enable_checkpointing=ckpt,check_val_every_n_epoch=5,
        precision=precision,resume_from_checkpoint=ckpt_path,
        auto_scale_batch_size="power" if args.batch_size == "tune" else None,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val)
    if strategy is None and args.batch_size == "tune":
        bs = trainer.tune(ssl,scale_batch_size_kwargs={"steps_per_trial":2,
                                                       "init_val":16})
    
    torch.cuda.empty_cache()
    force_cudnn_initialization()
    trainer.fit(ssl,val_dataloaders=train_loader)
    
    print("Validating...")
    test_metrics = trainer.test(ssl,train_loader)[0]
    for k in test_metrics:
        out = test_metrics[k]
        try:
            value = float(out.detach().numpy())
        except Exception:
            value = float(out)
        x = "{},{},{},{}".format(k,0,0,value)
        output_file.write(x+'\n')
    x = "{},{},{},{}".format(
        "train_ids",0,0,':'.join(train_pids))
    output_file.write(x+'\n')

    # just in case
    torch.cuda.empty_cache()

    output_file.close()

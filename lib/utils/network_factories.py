import numpy as np
import torch
import torch.nn.functional as F
from ..modules.layers.adn_fn import get_adn_fn
# classification
from ..utils.batch_preprocessing import BatchPreprocessing
from ..modules.classification.classification import (TabularClassifier)
from ..modules.classification.pl import (
    UNetEncoderPL,
    ClassNetPL,
    ViTClassifierPL,
    FactorizedViTClassifierPL,
    HybridClassifierPL)
# detection
from ..modules.object_detection.losses import complete_iou_loss
from ..modules.object_detection.pl import YOLONet3dPL
from ..utils import get_loss_param_dict,loss_factory
# segmentation
from lib.modules.segmentation.pl import (
    UNetPL,
    UNetPlusPlusPL,
    BrUNetPL,
    UNETRPL,
    SWINUNetPL
    )
# self-supervised learning
from lib.modules.self_supervised.pl import (
    SelfSLResNetPL,SelfSLUNetPL,
    SelfSLConvNeXtPL,IJEPAPL,
    ConvNeXt,ResNet,UNet,IJEPA)

from typing import Dict,Any,List,Callable

def get_classification_network(net_type:str,
                               network_config:Dict[str,Any],
                               dropout_param:float,
                               seed:int,
                               n_classes:int,
                               keys:List[str],
                               clinical_feature_keys:List[str],
                               train_loader_call:Callable,
                               max_epochs:int,
                               warmup_steps:int,
                               start_decay:int,
                               crop_size:int,
                               clinical_feature_means:torch.Tensor=None,
                               clinical_feature_stds:torch.Tensor=None,
                               label_smoothing=None,
                               mixup_alpha=None,
                               partial_mixup=None)->torch.nn.Module:
    if net_type == "unet":
        act_fn = network_config["activation_fn"]
    else:
        act_fn = "swish"
    adn_fn = get_adn_fn(3,"identity",act_fn=act_fn,
                        dropout_param=dropout_param)
    batch_preprocessing = BatchPreprocessing(
        label_smoothing,mixup_alpha,partial_mixup,seed)
    boilerplate_args = {
        "n_channels":len(keys),
        "n_classes":n_classes,
        "training_dataloader_call":train_loader_call,
        "image_key":"image",
        "label_key":"label",
        "n_epochs":max_epochs,
        "warmup_steps":warmup_steps,
        "training_batch_preproc":batch_preprocessing,
        "start_decay":start_decay}
    if net_type == "unet":
        network = UNetEncoderPL(
            head_structure=[
                network_config["depth"][-1] for _ in range(3)],
            head_adn_fn=get_adn_fn(
                1,"batch",act_fn="gelu",
                dropout_param=dropout_param),
            **boilerplate_args,
            **network_config)
    elif "vit" in net_type:
        image_size = [int(x) for x in crop_size]
        network_config["image_size"] = image_size
        if net_type == "vit":
            network = ViTClassifierPL(
                adn_fn=get_adn_fn(
                    1,"identity",act_fn="gelu",
                    dropout_param=dropout_param),
                **boilerplate_args,
                **network_config)
        elif net_type == "factorized_vit":
            for k in ["embed_method"]:
                if k in network_config:
                    del network_config[k]
            network = FactorizedViTClassifierPL(
                adn_fn=get_adn_fn(
                    1,"identity",act_fn="gelu",
                    dropout_param=dropout_param),
                **boilerplate_args,
                **network_config)                    
        
    else:
        network = ClassNetPL(
            net_type=net_type,adn_fn=adn_fn,
            **boilerplate_args,
            **network_config)
    
    if len(clinical_feature_keys) > 0:
        boilerplate_args_hybrid = {
            "training_dataloader_call":train_loader_call,
            "image_key":"image",
            "label_key":"label",
            "n_epochs":max_epochs,
            "warmup_steps":warmup_steps,
            "training_batch_preproc":batch_preprocessing,
            "start_decay":start_decay}
        
        for k in ["learning_rate","batch_size",
                  "loss_fn","loss_params"]:
            if k in network_config:
                boilerplate_args_hybrid[k] = network_config[k]
        tab_network = TabularClassifier(len(clinical_feature_keys),
                                        mlp_structure=[],
                                        mlp_adn_fn=torch.nn.Identity,
                                        n_classes=n_classes,
                                        feature_means=clinical_feature_means,
                                        feature_stds=clinical_feature_stds)
        network = HybridClassifierPL(
            convolutional_module=network,
            tabular_module=tab_network,
            **boilerplate_args_hybrid)

    return network

def get_detection_network(net_type:str,
                          network_config:Dict[str,Any],
                          dropout_param:float,
                          loss_gamma:float,
                          loss_comb:float,
                          class_weights:torch.Tensor,
                          train_loader_call:Callable,
                          iou_threshold:float,
                          n_classes:int,
                          anchor_array:np.ndarray,
                          n_epochs:int,
                          warmup_steps:int,
                          boxes_key:str,
                          box_class_key:str,
                          dev:str)->torch.nn.Module:
    if "activation_fn" in network_config:
        act_fn = network_config["activation_fn"]
    else:
        act_fn = "swish"

    if "classification_loss_fn" in network_config:
        k = "binary" if n_classes == 2 else "categorical"
        classification_loss_fn = loss_factory[k][
            network_config["classification_loss_fn"]]
    else:
        if n_classes == 2:
            classification_loss_fn = F.binary_cross_entropy
        else:
            classification_loss_fn = F.cross_entropy
    
    if "object_loss_fn" in network_config:
        object_loss_key = network_config["object_loss_fn"]
        object_loss_fn = loss_factory["binary"][
            network_config["object_loss_fn"]]
    else:
        object_loss_fn = F.binary_cross_entropy

    net_cfg = {k:network_config[k] for k in network_config
               if k not in ["activation_fn",
                            "classification_loss_fn",
                            "object_loss_fn"]}

    if "batch_size" not in net_cfg:
        net_cfg["batch_size"] = 1
        
    classification_loss_params = {}
    if (loss_gamma is None) or (loss_comb is None) or (class_weights is None):
        object_loss_params = {}
    else:
        object_loss_params = get_loss_param_dict(
            1.0,loss_gamma,loss_comb,0.5)[object_loss_key]

    adn_fn = get_adn_fn(
        3,norm_fn="batch",
        act_fn=act_fn,dropout_param=dropout_param)
    network = YOLONet3dPL(
        training_dataloader_call=train_loader_call,
        image_key="image",label_key="bb_map",boxes_key=boxes_key,
        box_label_key=box_class_key,
        anchor_sizes=anchor_array,adn_fn=adn_fn,
        iou_threshold=iou_threshold,
        classification_loss_fn=classification_loss_fn,
        object_loss_fn=object_loss_fn,
        reg_loss_fn=complete_iou_loss,
        object_loss_params=object_loss_params,
        classification_loss_params=classification_loss_params,
        n_epochs=n_epochs,warmup_steps=warmup_steps,
        n_classes=n_classes,
        **net_cfg)

    return network

def get_segmentation_network(net_type:str,
                             network_config:Dict[str,Any],
                             loss_params:Dict[str,Any],
                             bottleneck_classification:bool,
                             clinical_feature_keys:List[str],
                             all_aux_keys:List[str],
                             clinical_feature_params:Dict[str,torch.Tensor],
                             clinical_feature_key_net:str,
                             aux_key_net:str,
                             max_epochs:int,
                             encoding_operations:List[torch.nn.Module],
                             picai_eval:bool,
                             lr_encoder:float,
                             cosine_decay:bool,
                             encoder_checkpoint:str,
                             res_config_file:str,
                             deep_supervision:bool,
                             n_classes:int,
                             keys:List[str],
                             train_loader_call:Callable,
                             random_crop_size:List[int],
                             crop_size:List[int],
                             pad_size:List[int],
                             resize_size:List[int])->torch.nn.Module:
    
    def get_size(*size_list):
        for size in size_list:
            if size is not None:
                return size

    size = get_size(random_crop_size,
                    crop_size,
                    pad_size,
                    resize_size)

    boilerplate = dict(
        training_dataloader_call=train_loader_call,
        label_key="mask",
        loss_params=loss_params,
        n_classes=n_classes,
        bottleneck_classification=bottleneck_classification,
        skip_conditioning=len(all_aux_keys),
        skip_conditioning_key=aux_key_net,
        feature_conditioning=len(clinical_feature_keys),
        feature_conditioning_params=clinical_feature_params,
        feature_conditioning_key=clinical_feature_key_net,
        n_epochs=max_epochs,
        picai_eval=picai_eval,
        lr_encoder=lr_encoder,
        cosine_decay=cosine_decay
    )

    if net_type == "brunet":
        nc = network_config["n_channels"]
        network_config["n_channels"] = nc // len(keys)
        unet = BrUNetPL(
            encoders=encoding_operations,
            image_keys=keys,
            n_input_branches=len(keys),
            deep_supervision=deep_supervision,
            **boilerplate,
            **network_config)
        if encoder_checkpoint is not None and res_config_file is None:
            for encoder,ckpt in zip(unet.encoders,encoder_checkpoint):
                encoder.load_state_dict(torch.load(ckpt)["state_dict"])

    elif net_type == "unetpp":
        encoding_operations = encoding_operations[0]
        unet = UNetPlusPlusPL(
            encoding_operations=encoding_operations,
            image_key="image",
            **boilerplate,
            **network_config)

    elif net_type == "unet":
        encoding_operations = encoding_operations[0]
        unet = UNetPL(
            encoding_operations=encoding_operations,
            image_key="image",
            deep_supervision=deep_supervision,
            **boilerplate,
            **network_config)

    elif net_type == "unetr":
        sd = network_config["spatial_dimensions"]
        network_config["image_size"] = size[:sd]
        network_config["patch_size"] = network_config["patch_size"][:sd]
        unet = UNETRPL(
            image_key="image",
            deep_supervision=deep_supervision,
            **boilerplate,
            **network_config)

    elif net_type == "swin":
        sd = network_config["spatial_dimensions"]
        network_config["image_size"] = size[:sd]
        unet = SWINUNetPL(
            image_key="image",
            deep_supervision=deep_supervision,
            **boilerplate,
            **network_config)

    return unet

def get_ssl_network(train_loader_call:Callable,
                    max_epochs:int,
                    max_steps_optim:int,
                    warmup_steps:int,
                    ssl_method:str,
                    ema:torch.nn.Module,
                    net_type:str,
                    network_config_correct:Dict[str,Any],
                    stop_gradient:bool):
    if ssl_method == "ijepa":
        boilerplate = {
            "training_dataloader_call":train_loader_call,
            "image_key":"image",
            "n_epochs":max_epochs,
            "n_steps":max_steps_optim,
            "warmup_steps":warmup_steps,
            "ssl_method":ssl_method, # redundant but helpful for compatibility
            "ema":ema,
            "stop_gradient":stop_gradient,
            "temperature":0.1
        }
        ssl = IJEPAPL(**boilerplate,**network_config_correct)

    else:
        if ssl_method == "simclr":
            # simclr only uses a projection head, no prediction head
            del network_config_correct["prediction_head_args"]
        boilerplate = {
            "training_dataloader_call":train_loader_call,
            "aug_image_key_1":"augmented_image_1",
            "aug_image_key_2":"augmented_image_2",
            "box_key_1":"box_1",
            "box_key_2":"box_2",
            "n_epochs":max_epochs,
            "n_steps":max_steps_optim,
            "warmup_steps":warmup_steps,
            "ssl_method":ssl_method,
            "ema":ema,
            "stop_gradient":stop_gradient,
            "temperature":0.1}
        if net_type == "unet_encoder":
            ssl = SelfSLUNetPL(**boilerplate,**network_config_correct)
        elif net_type == "convnext":
            network_config_correct["backbone_args"] = {
                k:network_config_correct["backbone_args"][k] 
                for k in network_config_correct["backbone_args"]
                if k not in ["res_type"]}
            ssl = SelfSLConvNeXtPL(**boilerplate,**network_config_correct)
        else:
            ssl = SelfSLResNetPL(**boilerplate,**network_config_correct)

    return ssl

def get_ssl_network_no_pl(ssl_method:str,
                          net_type:str,
                          network_config_correct:Dict[str,Any]):
    if ssl_method == "ijepa":
        ssl = IJEPA(**network_config_correct)

    else:
        if net_type == "unet_encoder":
            ssl = UNet(**network_config_correct)
        elif net_type == "convnext":
            network_config_correct["backbone_args"] = {
                k:network_config_correct["backbone_args"][k] 
                for k in network_config_correct["backbone_args"]
                if k not in ["res_type"]}
            ssl = ConvNeXt(**network_config_correct)
        else:
            ssl = ResNet(**network_config_correct)

    return ssl
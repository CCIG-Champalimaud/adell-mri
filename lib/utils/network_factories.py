import numpy as np
import torch
import torch.functional as F
from ..modules.layers.adn_fn import get_adn_fn
# classification
from ..utils.batch_preprocessing import BatchPreprocessing
from ..modules.classification.classification import (TabularClassifier)
from ..modules.classification.pl import (
    UNetEncoderPL,FactorizedViTClassifierPL,ViTClassifierPL,ClassNetPL,
    HybridClassifierPL)
# detection
from ..modules.losses import complete_iou_loss
from ..modules.object_detection.pl import YOLONet3dPL
from ..utils import get_loss_param_dict,loss_factory

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
                          anchor_array:np.ndarray,
                          dev:str)->torch.nn.Module:
    if "activation_fn" in network_config:
        act_fn = network_config["activation_fn"]
    else:
        act_fn = "swish"

    if "classification_loss_fn" in network_config:
        class_loss_key = network_config["classification_loss_fn"]
        k = "binary"
        classification_loss_fn = loss_factory[k][
            network_config["classification_loss_fn"]]
    else:
        classification_loss_fn = F.binary_cross_entropy
    
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
        
    object_loss_params = get_loss_param_dict(
        1.0,loss_gamma,loss_comb,0.5)[object_loss_key]
    classification_loss_params = get_loss_param_dict(
        class_weights,loss_gamma,loss_comb,0.5)[class_loss_key]

    adn_fn = get_adn_fn(
        3,norm_fn="batch",
        act_fn=act_fn,dropout_param=dropout_param)
    network = YOLONet3dPL(
        training_dataloader_call=train_loader_call,
        image_key="image",label_key="bb_map",boxes_key="boxes",
        box_label_key="labels",
        anchor_sizes=anchor_array,
        n_c=2,adn_fn=adn_fn,iou_threshold=iou_threshold,
        classification_loss_fn=classification_loss_fn,
        object_loss_fn=object_loss_fn,
        reg_loss_fn=complete_iou_loss,
        classification_loss_params=classification_loss_params,
        object_loss_params=object_loss_params,
        **net_cfg)

    return network
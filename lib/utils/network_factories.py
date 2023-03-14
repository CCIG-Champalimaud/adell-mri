import torch
from ..modules.layers.adn_fn import get_adn_fn
from ..utils.batch_preprocessing import BatchPreprocessing
from ..modules.classification.classification import (TabularClassifier)
from ..modules.classification.pl import (
    UNetEncoderPL,FactorizedViTClassifierPL,ViTClassifierPL,ClassNetPL,
    HybridClassifierPL)

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
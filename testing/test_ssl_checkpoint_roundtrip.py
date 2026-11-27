import os
import sys

import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from adell_mri.modules.layers.conv_next import ConvNeXtBackbone
from adell_mri.modules.layers.res_net import ResNetBackbone
from adell_mri.modules.layers.vit import ViT
from adell_mri.modules.segmentation.unet import UNet
from adell_mri.modules.self_supervised.autoencoders import ViTMaskedAutoEncoder
from adell_mri.modules.self_supervised.pl import (
    SelfSLConvNeXtPL,
    SelfSLResNetPL,
)
from adell_mri.utils.network_factories import (
    convert_ssl_checkpoint,
    get_classification_network,
    initialise_ssl_from_ckpt,
)
from adell_mri.utils.pl_callbacks import ModelCheckpointWithMetadata

FAMILIES = ["resnet", "convnext", "vit", "mae"]
INPUT_SHAPE = (2, 1, 16, 16, 16)


def _build_encoder(family: str):
    """
    Builds a small encoder and the corresponding SSL network configuration.

    Args:
        family (str): one of "resnet", "convnext" or "vit".

    Returns:
        tuple: encoder, network config, state dict prefix and SSL method.
    """
    if family == "resnet":
        structure = [(8, 8, 3, 1), (16, 16, 3, 1), (32, 32, 3, 1)]
        encoder = ResNetBackbone(3, 1, structure)
        config = {
            "backbone_args": {
                "spatial_dim": 3,
                "in_channels": 1,
                "structure": structure,
            }
        }
        return encoder, config, "backbone.", "byol"
    if family == "convnext":
        structure = [(8, 8, 3, 1), (16, 16, 3, 1), (32, 32, 3, 1)]
        encoder = ConvNeXtBackbone(3, 1, structure, first_layer_stride=2)
        config = {
            "backbone_args": {
                "spatial_dim": 3,
                "in_channels": 1,
                "structure": structure,
                "first_layer_stride": 2,
            }
        }
        return encoder, config, "backbone.", "byol"
    if family == "mae":
        encoder_args = {
            "image_size": [16, 16, 16],
            "patch_size": [4, 4, 4],
            "in_channels": 1,
            "embed_dim": 32,
            "number_of_blocks": 4,
            "hidden_dim": 64,
            "n_heads": 2,
            "mlp_structure": [64],
        }
        decoder_args = {
            "number_of_blocks": 2,
            "hidden_dim": 64,
            "n_heads": 2,
            "mlp_structure": [64],
        }
        encoder = ViTMaskedAutoEncoder(
            image_size=(16, 16, 16),
            patch_size=(4, 4, 4),
            in_channels=1,
            input_dim_size=32,
            encoder_args=encoder_args,
            decoder_args=decoder_args,
            mask_fraction=0.5,
        )
        config = {
            "encoder_args": encoder_args,
            "decoder_args": decoder_args,
            "mask_fraction": 0.5,
        }
        return encoder, config, "model.", "mae"
    encoder = ViT(
        image_size=(16, 16, 16),
        patch_size=(4, 4, 4),
        in_channels=1,
        number_of_blocks=6,
        attention_dim=32,
        embedding_size=32,
    )
    config = {
        "backbone_args": {
            "image_size": [16, 16, 16],
            "patch_size": [4, 4, 4],
            "in_channels": 1,
            "number_of_blocks": 6,
            "attention_dim": 32,
            "embedding_size": 32,
        }
    }
    return encoder, config, "encoder_.", "dino"


def _write_checkpoint(
    tmp_path,
    family: str,
    name: str = "ssl.ckpt",
    include_net_type: bool = True,
):
    """
    Writes an SSL checkpoint using the production checkpoint callback.

    Args:
        tmp_path: pytest temporary directory.
        family (str): encoder family.
        name (str, optional): checkpoint file name. Defaults to "ssl.ckpt".
        include_net_type (bool, optional): whether to store net_type and
            ssl_method in the metadata. Defaults to True.

    Returns:
        tuple: checkpoint path and the encoder used to build it.
    """
    encoder, config, prefix, ssl_method = _build_encoder(family)
    state_dict = {prefix + k: v for k, v in encoder.state_dict().items()}
    metadata = {"network_config": config}
    if include_net_type is True:
        metadata["net_type"] = family
        metadata["ssl_method"] = ssl_method
    callback = ModelCheckpointWithMetadata(metadata=metadata)
    checkpoint = {
        "state_dict": state_dict,
        "callbacks": {type(callback).__name__: callback.state_dict()},
        "epoch": 0,
        "global_step": 0,
    }
    path = tmp_path / name
    torch.save(checkpoint, path)
    return str(path), encoder


def _assert_same_weights(encoder: torch.nn.Module, loaded: torch.nn.Module):
    """
    Asserts that two modules have identical weights.

    Args:
        encoder (torch.nn.Module): reference module.
        loaded (torch.nn.Module): module loaded from a checkpoint.
    """
    reference = encoder.state_dict()
    loaded_state = loaded.state_dict()
    assert reference.keys() == loaded_state.keys()
    for key in reference:
        assert torch.equal(reference[key], loaded_state[key]), key


@pytest.mark.parametrize("family", FAMILIES)
def test_ssl_checkpoint_saving_and_loading(tmp_path, family):
    path, encoder = _write_checkpoint(tmp_path, family)

    loaded = initialise_ssl_from_ckpt(path)

    loaded_weights = getattr(loaded, "model", loaded)
    assert type(loaded_weights).__name__ == type(encoder).__name__
    _assert_same_weights(encoder, loaded_weights)


def _build_ssl_pl(family: str):
    """
    Builds a small SSL Lightning module.

    Args:
        family (str): either "resnet" or "convnext".

    Returns:
        tuple: the SSL module and its backbone arguments.
    """
    structure = [(8, 8, 3, 1), (16, 16, 3, 1), (32, 32, 3, 1)]
    projection_head_args = {
        "in_channels": 32,
        "structure": [32, 16],
        "adn_fn": torch.nn.Identity,
    }
    if family == "resnet":
        backbone_args = {
            "spatial_dim": 3,
            "in_channels": 1,
            "structure": structure,
            "adn_fn": torch.nn.Identity,
            "res_type": "resnet",
        }
        module = SelfSLResNetPL(
            backbone_args=backbone_args,
            projection_head_args=projection_head_args,
            prediction_head_args=None,
            ssl_method="simclr",
            training_dataloader_call=None,
        )
    else:
        backbone_args = {
            "spatial_dim": 3,
            "in_channels": 1,
            "structure": structure,
            "adn_fn": torch.nn.Identity,
        }
        module = SelfSLConvNeXtPL(
            backbone_args=backbone_args,
            projection_head_args=projection_head_args,
            prediction_head_args=None,
            ssl_method="simclr",
            training_dataloader_call=None,
        )
    return module, backbone_args


@pytest.mark.parametrize("family", ["resnet", "convnext"])
def test_ssl_pl_checkpoint_roundtrip(tmp_path, family):
    module, backbone_args = _build_ssl_pl(family)
    metadata = {
        "network_config": {"backbone_args": backbone_args},
        "net_type": family,
        "ssl_method": "simclr",
    }
    callback = ModelCheckpointWithMetadata(metadata=metadata)
    checkpoint = {
        "state_dict": module.state_dict(),
        "callbacks": {type(callback).__name__: callback.state_dict()},
        "epoch": 0,
        "global_step": 0,
    }
    path = str(tmp_path / "pl.ckpt")
    torch.save(checkpoint, path)

    loaded = initialise_ssl_from_ckpt(path)

    _assert_same_weights(module.backbone, loaded)


@pytest.mark.parametrize("family", FAMILIES)
def test_forward_with_intermediate(tmp_path, family):
    path, encoder = _write_checkpoint(tmp_path, family)
    loaded = initialise_ssl_from_ckpt(path)
    loaded.eval()

    bottleneck, intermediates = loaded.forward_with_intermediate(
        torch.rand(INPUT_SHAPE)
    )

    assert len(intermediates) == len(loaded.out_channels)
    assert bottleneck.shape[1] == loaded.out_channels[-1]
    for feature_map, channels in zip(intermediates, loaded.out_channels):
        assert feature_map.shape[1] == channels


def test_conversion_of_old_checkpoint(tmp_path):
    path, encoder = _write_checkpoint(
        tmp_path, "resnet", include_net_type=False
    )

    with pytest.raises(ValueError):
        initialise_ssl_from_ckpt(path)

    converted_path = convert_ssl_checkpoint(
        path, output_path=str(tmp_path / "converted.ckpt")
    )
    loaded = initialise_ssl_from_ckpt(converted_path)

    _assert_same_weights(encoder, loaded)


@pytest.mark.parametrize("family", ["resnet", "vit", "mae"])
def test_pretrained_encoder_segmentation(tmp_path, family):
    path, _ = _write_checkpoint(tmp_path, family)
    encoder = initialise_ssl_from_ckpt(path)
    depth = list(encoder.out_channels)

    unet = UNet(
        spatial_dimensions=3,
        encoder=encoder,
        depth=depth,
        kernel_sizes=[3 for _ in depth],
        strides=[2 for _ in depth],
        interpolation="trilinear",
    )
    unet.eval()

    output, _ = unet(torch.rand(INPUT_SHAPE))

    assert output.shape == (INPUT_SHAPE[0], 1, *INPUT_SHAPE[2:])


@pytest.mark.parametrize("family", ["resnet", "vit", "mae"])
@pytest.mark.parametrize("objective", ["cat", "ord"])
def test_pretrained_encoder_classification(tmp_path, family, objective):
    path, _ = _write_checkpoint(tmp_path, family)

    network = get_classification_network(
        net_type="pretrained_encoder",
        network_config={},
        dropout_param=0.0,
        seed=42,
        n_classes=3,
        keys=["image"],
        clinical_feature_keys=[],
        train_loader_call=None,
        max_epochs=1,
        warmup_steps=0,
        start_decay=1,
        crop_size=list(INPUT_SHAPE[2:]),
        encoder_checkpoint=path,
        objective=objective,
    )
    network.eval()

    output = network(torch.rand(INPUT_SHAPE))

    n_outputs = 3 if objective == "cat" else 2
    assert output.shape == (INPUT_SHAPE[0], n_outputs)

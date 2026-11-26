import pytest
import torch

from adell_mri.modules.layers.adn_fn import get_adn_fn
from adell_mri.modules.layers.conv_next import ConvNeXt
from adell_mri.modules.layers.res_net import ResNet
from adell_mri.modules.layers.vit import ViT, infer_return_at
from adell_mri.modules.segmentation.encoder_unet import EncoderUNet
from adell_mri.modules.segmentation.unet import UNet
from adell_mri.utils.encoder_loading import (
    build_encoder_manifest,
    load_ssl_encoder,
    read_encoder_manifest,
)
from adell_mri.utils.network_factories import (
    get_classification_network,
    get_deconfounded_classification_network,
    get_segmentation_network,
)


def _resnet_checkpoint() -> dict:
    backbone_args = {
        "spatial_dim": 2,
        "in_channels": 1,
        "structure": [[8, 8, 3, 1], [16, 16, 3, 1]],
        "maxpool_structure": [2, 2],
        "adn_fn": get_adn_fn(2, "batch", "relu", 0.0),
    }
    network = ResNet(
        backbone_args=backbone_args,
        projection_head_args={
            "in_channels": 16,
            "structure": [16, 8],
            "adn_fn": torch.nn.Identity,
        },
    )
    manifest = build_encoder_manifest(
        net_type="resnet",
        ssl_method="simclr",
        network_config={
            "backbone_args": backbone_args,
            "norm_fn": "batch",
            "act_fn": "relu",
        },
        spatial_dim=2,
        in_channels=1,
    )
    return {"state_dict": network.state_dict(), "encoder_config": manifest}


def _convnext_checkpoint() -> dict:
    backbone_args = {
        "spatial_dim": 2,
        "in_channels": 1,
        "structure": [[8, 8, 3, 1], [16, 16, 3, 1]],
        "maxpool_structure": [2, 2],
        "first_layer_stride": 4,
    }
    network = ConvNeXt(
        backbone_args=backbone_args,
        projection_head_args={
            "in_channels": 16,
            "structure": [16, 8],
            "adn_fn": torch.nn.Identity,
        },
    )
    manifest = build_encoder_manifest(
        net_type="convnext",
        ssl_method="simclr",
        network_config={
            "backbone_args": backbone_args,
            "norm_fn": "batch",
            "act_fn": "relu",
        },
        spatial_dim=2,
        in_channels=1,
    )
    return {"state_dict": network.state_dict(), "encoder_config": manifest}


def _vit_checkpoint() -> dict:
    backbone_args = {
        "image_size": [32, 32],
        "patch_size": [8, 8],
        "in_channels": 1,
        "number_of_blocks": 4,
        "attention_dim": 16,
        "hidden_dim": 16,
        "embedding_size": 16,
        "n_heads": 4,
        "use_class_token": True,
    }
    network = ViT(**backbone_args)
    manifest = build_encoder_manifest(
        net_type="vit",
        ssl_method="dino",
        network_config={
            "backbone_args": backbone_args,
            "norm_fn": "layer",
            "act_fn": "gelu",
        },
        spatial_dim=2,
        in_channels=1,
    )
    state_dict = {
        f"encoder_.{key}": value for key, value in network.state_dict().items()
    }
    return {"state_dict": state_dict, "encoder_config": manifest}


def _mae_checkpoint() -> dict:
    from adell_mri.modules.self_supervised.autoencoders import (
        ViTMaskedAutoEncoder,
    )

    encoder_args = {
        "number_of_blocks": 2,
        "hidden_dim": 16,
        "n_heads": 4,
        "mlp_structure": [32],
    }
    decoder_args = {
        "number_of_blocks": 1,
        "hidden_dim": 16,
        "n_heads": 4,
        "mlp_structure": [32],
    }
    network = ViTMaskedAutoEncoder(
        image_size=[32, 32],
        patch_size=[8, 8],
        in_channels=1,
        input_dim_size=16,
        encoder_args=encoder_args,
        decoder_args=decoder_args,
    )
    manifest = build_encoder_manifest(
        net_type="vit",
        ssl_method="mae",
        network_config={
            "image_size": [32, 32],
            "patch_size": [8, 8],
            "in_channels": 1,
            "input_dim_size": 16,
            "encoder_args": encoder_args,
            "decoder_args": decoder_args,
        },
        spatial_dim=2,
        in_channels=1,
    )
    state_dict = {
        f"model.{key}": value for key, value in network.state_dict().items()
    }
    return {"state_dict": state_dict, "encoder_config": manifest}


def _unet_checkpoint() -> dict:
    unet_args = {
        "spatial_dimensions": 2,
        "in_channels": 1,
        "depth": [8, 16, 32],
        "kernel_sizes": [3, 3, 3],
        "strides": [2, 2, 2],
        "encoder_only": True,
    }
    network = UNet(**unet_args)
    manifest = build_encoder_manifest(
        net_type="unet_encoder",
        ssl_method="simclr",
        network_config=unet_args,
        spatial_dim=2,
        in_channels=1,
    )
    return {"state_dict": network.state_dict(), "encoder_config": manifest}


@pytest.mark.parametrize(
    "checkpoint_factory,expected_family",
    [
        (_resnet_checkpoint, "resnet"),
        (_convnext_checkpoint, "convnext"),
        (_vit_checkpoint, "vit"),
        (_mae_checkpoint, "mae"),
        (_unet_checkpoint, "unet"),
    ],
)
def test_manifest_roundtrip(checkpoint_factory, expected_family):
    checkpoint = checkpoint_factory()
    manifest = read_encoder_manifest(checkpoint)
    assert manifest["family"] == expected_family
    encoder = load_ssl_encoder(checkpoint)
    x = torch.rand(2, 1, 32, 32)
    with torch.no_grad():
        bottleneck, skips = encoder.forward_with_intermediate(x)
    assert bottleneck.ndim == 4
    assert len(skips) > 0


def test_manifest_falls_back_to_hyper_parameters():
    checkpoint = _vit_checkpoint()
    checkpoint.pop("encoder_config")
    checkpoint["hyper_parameters"] = {
        "backbone_args": {
            "image_size": [32, 32],
            "patch_size": [8, 8],
            "in_channels": 1,
            "number_of_blocks": 4,
            "attention_dim": 16,
            "embedding_size": 16,
            "n_heads": 4,
            "use_class_token": True,
        }
    }
    manifest = read_encoder_manifest(checkpoint)
    assert manifest["family"] == "vit"
    load_ssl_encoder(checkpoint)


@pytest.mark.parametrize(
    "checkpoint_factory,expected_family,backbone_args",
    [
        (
            _resnet_checkpoint,
            "resnet",
            {
                "spatial_dim": 2,
                "in_channels": 1,
                "structure": [[8, 8, 3, 1], [16, 16, 3, 1]],
                "maxpool_structure": [2, 2],
                "res_type": "resnet",
            },
        ),
        (
            _convnext_checkpoint,
            "convnext",
            {
                "spatial_dim": 2,
                "in_channels": 1,
                "structure": [[8, 8, 3, 1], [16, 16, 3, 1]],
                "maxpool_structure": [2, 2],
                "first_layer_stride": 4,
            },
        ),
    ],
)
def test_manifest_infers_convolutional_family_from_weights(
    checkpoint_factory, expected_family, backbone_args
):
    checkpoint = checkpoint_factory()
    checkpoint.pop("encoder_config")
    checkpoint["hyper_parameters"] = {"backbone_args": backbone_args}
    manifest = read_encoder_manifest(checkpoint)
    assert manifest["family"] == expected_family
    load_ssl_encoder(checkpoint)


def test_infer_return_at_is_evenly_spaced_and_in_bounds():
    indices = infer_return_at(12, 4)
    assert indices == sorted(indices)
    assert indices[0] == 0
    assert max(indices) < 12
    assert len(indices) == 4


@pytest.mark.parametrize(
    "checkpoint_factory,spatial_dimensions,input_shape",
    [
        (_resnet_checkpoint, 2, [32, 32]),
        (_convnext_checkpoint, 2, [32, 32]),
        (_vit_checkpoint, 2, [32, 32]),
        (_unet_checkpoint, 2, [32, 32]),
    ],
)
def test_encoder_unet_forward(
    checkpoint_factory, spatial_dimensions, input_shape
):
    encoder = load_ssl_encoder(checkpoint_factory())
    model = EncoderUNet(
        encoder=encoder,
        encoder_in_channels=1,
        input_shape=input_shape,
        spatial_dimensions=spatial_dimensions,
        in_channels=1,
        depth=[8, 16, 32, 32],
        kernel_sizes=[3, 3, 3, 3],
        n_classes=2,
        upscale_type="transpose",
        link_type="identity",
        norm_type="batch",
        activation_fn=torch.nn.ReLU,
        conv_type="regular",
        padding=1,
    )
    x = torch.rand(2, 1, *input_shape)
    output, _, bottleneck = model(x, return_features=True)
    assert output.shape == (2, 1, *input_shape)
    assert bottleneck.ndim == len(input_shape) + 2


def test_segmentation_encoder_unet_factory():
    network = get_segmentation_network(
        net_type="encoder_unet",
        network_config={
            "spatial_dimensions": 2,
            "in_channels": 1,
            "depth": [8, 16, 32],
            "kernel_sizes": [3, 3, 3],
            "loss_fn": torch.nn.BCEWithLogitsLoss(),
            "batch_size": 2,
            "learning_rate": 1e-4,
            "conv_type": "regular",
            "link_type": "identity",
            "upscale_type": "transpose",
            "norm_type": "batch",
            "activation_fn": torch.nn.ReLU,
            "padding": 1,
        },
        bottleneck_classification=False,
        clinical_feature_keys=[],
        all_aux_keys=[],
        clinical_feature_params=None,
        clinical_feature_key_net=None,
        aux_key_net=None,
        max_epochs=1,
        encoding_operations=[None],
        picai_eval=False,
        lr_encoder=None,
        encoder_checkpoint=[_resnet_checkpoint()],
        deep_supervision=False,
        n_classes=2,
        keys=["image"],
        optimizer_str="adamw",
        train_loader_call=None,
        random_crop_size=[32, 32],
        crop_size=None,
        pad_size=None,
        resize_size=None,
        semi_supervised=False,
        max_steps_optim=10,
    )
    output = network(torch.rand(2, 1, 32, 32))
    assert output[0].shape == (2, 1, 32, 32)


def test_classification_cat_uses_ssl_encoder():
    network = get_classification_network(
        net_type="cat",
        network_config={
            "spatial_dimensions": 2,
            "loss_fn": torch.nn.BCEWithLogitsLoss(),
            "classification_structure": [16, 8],
            "batch_size": 2,
            "learning_rate": 1e-4,
        },
        dropout_param=0.0,
        seed=42,
        n_classes=2,
        keys=["image"],
        clinical_feature_keys=[],
        train_loader_call=None,
        max_epochs=1,
        warmup_steps=0,
        start_decay=1,
        crop_size=[32, 32],
        encoder_checkpoint=[_resnet_checkpoint()],
    )
    output = network(torch.rand(2, 1, 32, 32))
    assert output.shape == (2, 1)


def test_classification_vit_uses_ssl_encoder():
    network = get_classification_network(
        net_type="vit",
        network_config={
            "loss_fn": torch.nn.BCEWithLogitsLoss(),
            "batch_size": 2,
            "learning_rate": 1e-4,
        },
        dropout_param=0.0,
        seed=42,
        n_classes=2,
        keys=["image"],
        clinical_feature_keys=[],
        train_loader_call=None,
        max_epochs=1,
        warmup_steps=0,
        start_decay=1,
        crop_size=[32, 32],
        encoder_checkpoint=_vit_checkpoint(),
    )
    output = network(torch.rand(2, 1, 32, 32))
    assert output.shape == (2, 1)


@pytest.mark.parametrize(
    "net_type,checkpoint_factory",
    [
        ("unet", _resnet_checkpoint),
        ("vit", _resnet_checkpoint),
        ("vgg", _convnext_checkpoint),
        ("cat", _vit_checkpoint),
        ("ord", _unet_checkpoint),
    ],
)
def test_classification_accepts_any_encoder(net_type, checkpoint_factory):
    network = get_classification_network(
        net_type=net_type,
        network_config={
            "spatial_dimensions": 2,
            "loss_fn": torch.nn.BCEWithLogitsLoss(),
            "classification_structure": [16, 8],
            "batch_size": 2,
            "learning_rate": 1e-4,
            "activation_fn": torch.nn.ReLU,
            "depth": [8, 16, 32],
            "kernel_sizes": [3, 3, 3],
            "strides": [2, 2, 2],
        },
        dropout_param=0.0,
        seed=42,
        n_classes=2,
        keys=["image"],
        clinical_feature_keys=[],
        train_loader_call=None,
        max_epochs=1,
        warmup_steps=0,
        start_decay=1,
        crop_size=[32, 32],
        encoder_checkpoint=checkpoint_factory(),
    )
    assert network(torch.rand(2, 1, 32, 32)).shape == (2, 1)


@pytest.mark.parametrize("lr_encoder", [0, 1e-5])
def test_classification_encoder_learning_rate(lr_encoder):
    network = get_classification_network(
        net_type="cat",
        network_config={
            "spatial_dimensions": 2,
            "loss_fn": torch.nn.BCEWithLogitsLoss(),
            "classification_structure": [16, 8],
            "batch_size": 2,
            "learning_rate": 1e-4,
        },
        dropout_param=0.0,
        seed=42,
        n_classes=2,
        keys=["image"],
        clinical_feature_keys=[],
        train_loader_call=None,
        max_epochs=1,
        warmup_steps=0,
        start_decay=1,
        crop_size=[32, 32],
        encoder_checkpoint=_resnet_checkpoint(),
        lr_encoder=lr_encoder,
    )
    network.configure_optimizers()
    encoder_parameters = [
        parameter
        for name, parameter in network.named_parameters()
        if "feature_extraction" in name
    ]
    assert len(encoder_parameters) > 0
    if lr_encoder == 0:
        assert all(not p.requires_grad for p in encoder_parameters)
    else:
        assert all(p.requires_grad for p in encoder_parameters)


def test_deconfounded_classification_accepts_vit_encoder():
    network = get_deconfounded_classification_network(
        network_config={
            "spatial_dimensions": 2,
            "loss_fn": torch.nn.BCEWithLogitsLoss(),
            "classification_structure": [16, 8],
            "batch_size": 2,
            "learning_rate": 1e-4,
            "activation_fn": torch.nn.ReLU,
            "depth": [8, 16, 32],
            "kernel_sizes": [3, 3, 3],
            "strides": [2, 2, 2],
        },
        dropout_param=0.0,
        seed=42,
        n_classes=2,
        keys=["image"],
        cat_confounder_key=None,
        cont_confounder_key=None,
        cat_vars=None,
        cont_vars=0,
        train_loader_call=None,
        max_epochs=1,
        warmup_steps=0,
        start_decay=1,
        encoder_checkpoint=_vit_checkpoint(),
        lr_encoder=0,
    )
    output = network(torch.rand(2, 1, 32, 32))
    assert output[0].shape == (2, 1)


def test_encoder_unet_input_adapter():
    encoder = load_ssl_encoder(_convnext_checkpoint())
    model = EncoderUNet(
        encoder=encoder,
        encoder_in_channels=1,
        input_shape=[32, 32],
        spatial_dimensions=2,
        in_channels=2,
        depth=[8, 16, 32, 32],
        kernel_sizes=[3, 3, 3, 3],
        n_classes=2,
        upscale_type="transpose",
        link_type="identity",
        norm_type="batch",
        activation_fn=torch.nn.ReLU,
    )
    x = torch.rand(1, 2, 32, 32)
    output, _ = model(x)
    assert output.shape == (1, 1, 32, 32)

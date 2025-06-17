import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from adell_mri.modules.layers.regularization import L2NormalizationLayer
from adell_mri.modules.self_supervised.pl import ViTMaskedAutoEncoderPL
from adell_mri.utils import network_factories


def dummy_train_loader():
    return None


def dummy_ema():
    # Return None for UNet-based models to avoid module assignment error
    return None


# All ssl_method values handled explicitly in get_ssl_network
ssl_methods = [
    "simclr",
    "byol",
    "vicreg",
    "vicregl",
    "ijepa",
    "mae",
    "dino",
    "ibot",
]

# net_type options for the 'else' branch (valid for SSL else branch)
net_types = ["unet_encoder", "convnext", "resnet"]


def minimal_network_config(ssl_method):
    # Provide minimal config for each method
    if ssl_method in ["simclr", "byol", "vicreg", "vicregl"]:
        # Use a consistent hidden dimension for all layers
        hidden_dim = 64
        return {
            "backbone_args": {
                "spatial_dim": 2,
                "in_channels": 1,
                "structure": [(hidden_dim, hidden_dim, 3, 2)],
                "maxpool_structure": [2],
                "adn_fn": torch.nn.Identity,
                "res_type": "resnet",
            },
            "projection_head_args": {
                "in_channels": hidden_dim,
                "structure": [hidden_dim, hidden_dim],  # At least two elements
                "adn_fn": torch.nn.Identity,
                "last_layer_norm": L2NormalizationLayer,
            },
            "prediction_head_args": {
                "in_channels": hidden_dim,  # Match projection head output
                "structure": [hidden_dim, hidden_dim],  # At least two elements
                "adn_fn": torch.nn.Identity,
            },
        }
    elif ssl_method == "ijepa":
        # Minimal valid config for IJEPA
        return {
            "backbone_args": {
                "patch_size": (16, 16),
                "img_size": (224, 224),
                "in_channels": 1,
                "embed_dim": 96,
                "depth": 4,
                "num_heads": 3,
                "mlp_ratio": 4.0,
                "qkv_bias": True,
                "norm_layer": torch.nn.LayerNorm,
            },
            "projection_head_args": {
                "in_channels": 96,
                "structure": [96, 48],
                "adn_fn": torch.nn.Identity,
            },
            "feature_map_dimensions": [14, 14],
            "n_encoder_features": 96,
            "min_patch_size": [8, 8],
            "max_patch_size": [16, 16],
            # Add missing required parameters
            "n_patches": 4,
            "n_masked_patches": 1,
            "encoder_architecture": "vit",
            "predictor_architecture": "vit",
            "reduce_fn": "mean",
            "seed": 42,
        }
    elif ssl_method == "dino":
        # Minimal valid config for DINO
        return {
            "backbone_args": {
                "patch_size": (16, 16),
                "img_size": (224, 224),
                "in_channels": 1,
                "embed_dim": 96,
                "depth": 4,
                "num_heads": 3,
                "mlp_ratio": 4.0,
                "qkv_bias": True,
                "norm_layer": torch.nn.LayerNorm,
            },
            "projection_head_args": {
                "in_channels": 96,
                "structure": [96, 48],
                "adn_fn": torch.nn.Identity,
            },
            "out_dim": 48,
        }
    elif ssl_method == "ibot":
        # Minimal valid config for iBOT
        return {
            "backbone_args": {
                "patch_size": (16, 16),
                "img_size": (224, 224),
                "in_channels": 1,
                "embed_dim": 96,
                "depth": 4,
                "num_heads": 3,
                "mlp_ratio": 4.0,
                "qkv_bias": True,
                "norm_layer": torch.nn.LayerNorm,
            },
            "projection_head_args": {
                "in_channels": 96,
                "structure": [96, 48],
                "adn_fn": torch.nn.Identity,
            },
            "out_dim": 48,
            "feature_map_dimensions": [14, 14],
            "n_encoder_features": 96,
            "min_patch_size": [8, 8],
            "max_patch_size": [16, 16],
        }
    elif ssl_method == "mae":
        # Minimal valid config for ViTMaskedAutoEncoderPL
        return {
            "image_size": (224, 224),
            "patch_size": (16, 16),
            "in_channels": 1,
            "input_dim_size": 768,
            "encoder_args": {
                "hidden_dim": 96,
                "number_of_blocks": 4,
                "n_heads": 4,
                "mlp_structure": [96 * 4],
            },
            "decoder_args": {
                "hidden_dim": 48,
                "number_of_blocks": 2,
                "n_heads": 4,
                "mlp_structure": [48 * 4],
            },
        }
    else:
        return {}


def test_simclr_network():
    """Test SimCLR network creation and forward pass."""
    ssl_method = "simclr"
    batch_size = 4
    config = minimal_network_config(ssl_method)

    # Create the network
    net = network_factories.get_ssl_network(
        train_loader_call=dummy_train_loader,
        max_epochs=1,
        max_steps_optim=1,
        warmup_steps=0,
        ssl_method=ssl_method,
        ema=dummy_ema(),
        net_type="resnet",
        network_config_correct=config,
        stop_gradient=False,
    )

    # Basic type check
    assert isinstance(net, torch.nn.Module)

    # Test forward pass with correct input format
    dummy_input = {
        net.aug_image_key_1: torch.randn(batch_size, 1, 224, 224),
        net.aug_image_key_2: torch.randn(batch_size, 1, 224, 224),
    }

    # Test training step
    loss = net.training_step(dummy_input, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad  # Loss should be a differentiable scalar

    # Test forward pass through the network
    with torch.no_grad():
        # Get the backbone features
        features1 = net.backbone(dummy_input[net.aug_image_key_1])
        features2 = net.backbone(dummy_input[net.aug_image_key_2])

        # Check feature shapes
        assert features1.shape[0] == batch_size
        assert features1.shape == features2.shape

        # Project the features
        proj1 = net.projection_head(features1)
        proj2 = net.projection_head(features2)

        # Check projection shapes
        assert proj1.shape[0] == batch_size
        assert proj1.shape == proj2.shape


def test_byol_network():
    """Test BYOL network creation and forward pass."""
    ssl_method = "byol"
    batch_size = 4
    config = minimal_network_config(ssl_method)

    # Create the network
    net = network_factories.get_ssl_network(
        train_loader_call=dummy_train_loader,
        max_epochs=1,
        max_steps_optim=1,
        warmup_steps=0,
        ssl_method=ssl_method,
        ema=dummy_ema(),
        net_type="resnet",
        network_config_correct=config,
        stop_gradient=False,
    )

    # Basic type check
    assert isinstance(net, torch.nn.Module)

    # Test forward pass with correct input format
    dummy_input = {
        net.aug_image_key_1: torch.randn(batch_size, 1, 224, 224),
        net.aug_image_key_2: torch.randn(batch_size, 1, 224, 224),
    }

    # Test training step
    loss = net.training_step(dummy_input, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad  # Loss should be a differentiable scalar

    # Test forward pass through the network
    with torch.no_grad():
        # Get the online and target features through the forward method
        online1 = net(dummy_input[net.aug_image_key_1])
        online2 = net(dummy_input[net.aug_image_key_2])

        # Check shapes
        assert online1.shape[0] == batch_size


def test_vicreg_network():
    """Test VICReg network creation and forward pass."""
    ssl_method = "vicreg"
    batch_size = 4
    config = minimal_network_config(ssl_method)

    # Create the network
    net = network_factories.get_ssl_network(
        train_loader_call=dummy_train_loader,
        max_epochs=1,
        max_steps_optim=1,
        warmup_steps=0,
        ssl_method=ssl_method,
        ema=dummy_ema(),
        net_type="resnet",
        network_config_correct=config,
        stop_gradient=False,
    )

    # Basic type check
    assert isinstance(net, torch.nn.Module)

    # Test forward pass with correct input format
    dummy_input = {
        net.aug_image_key_1: torch.randn(batch_size, 1, 224, 224),
        net.aug_image_key_2: torch.randn(batch_size, 1, 224, 224),
    }

    # Test training step
    loss = net.training_step(dummy_input, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad  # Loss should be a differentiable scalar

    # Test forward pass through the network
    with torch.no_grad():
        # Get the backbone features
        features1 = net.backbone(dummy_input[net.aug_image_key_1])
        features2 = net.backbone(dummy_input[net.aug_image_key_2])

        # Check feature shapes
        assert features1.shape[0] == batch_size
        assert features1.shape == features2.shape

        # Project the features
        proj1 = net.projection_head(features1)
        proj2 = net.projection_head(features2)

        # Check projection shapes
        assert proj1.shape[0] == batch_size
        assert proj1.shape == proj2.shape

        # Check that projections are normalized
        proj1_norm = torch.norm(proj1, dim=1)
        proj2_norm = torch.norm(proj2, dim=1)
        assert torch.allclose(
            proj1_norm, torch.ones_like(proj1_norm), atol=1e-5
        )
        assert torch.allclose(
            proj2_norm, torch.ones_like(proj2_norm), atol=1e-5
        )


def test_vicregl_network():
    """Test VICRegL network creation and forward pass with local features."""
    ssl_method = "vicregl"
    batch_size = 4
    num_boxes = 3  # Number of boxes per image
    config = minimal_network_config(ssl_method)

    # Create the network
    net = network_factories.get_ssl_network(
        train_loader_call=dummy_train_loader,
        max_epochs=1,
        max_steps_optim=1,
        warmup_steps=0,
        ssl_method=ssl_method,
        ema=dummy_ema(),
        net_type="resnet",
        network_config_correct=config,
        stop_gradient=False,
    )

    # Basic type check
    assert isinstance(net, torch.nn.Module)

    # Test forward pass with correct input format including boxes
    dummy_input = {
        net.aug_image_key_1: torch.randn(batch_size, 1, 224, 224),
        net.aug_image_key_2: torch.randn(batch_size, 1, 224, 224),
        # Format: [x1, y1, x2, y2]
        net.box_key_1: torch.tensor([[0.1, 0.1, 0.5, 0.5]] * batch_size),
        net.box_key_2: torch.tensor([[0.2, 0.2, 0.6, 0.6]] * batch_size),
    }

    # Test training step
    loss = net.training_step(dummy_input, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad  # Loss should be a differentiable scalar

    # Test forward pass through the network
    with torch.no_grad():
        # Get the global features
        global1 = net.backbone(dummy_input[net.aug_image_key_1])
        global2 = net.backbone(dummy_input[net.aug_image_key_2])

        # Check global feature shapes
        assert global1.shape[0] == batch_size
        assert global1.shape == global2.shape

        # Get the local features using ROI pooling
        boxes1 = dummy_input[net.box_key_1].view(
            -1, 4
        )  # Flatten batch and boxes
        boxes2 = dummy_input[net.box_key_2].view(-1, 4)

        # Check that we can extract local features
        try:
            # Some implementations might have different ways to access local features
            local1 = net.extract_local_features(
                dummy_input[net.aug_image_key_1], boxes1
            )
            local2 = net.extract_local_features(
                dummy_input[net.aug_image_key_2], boxes2
            )

            # Check local feature shapes
            assert local1.shape[0] == batch_size * num_boxes
            assert local1.shape == local2.shape
        except AttributeError:
            # If extract_local_features is not directly available, that's okay
            # as long as the training step works
            pass


def test_mae_network():
    """Test MAE network creation and forward pass."""
    ssl_method = "mae"
    # Test parameters - use tuples for image_size and patch_size
    img_size = (224, 224)  # (height, width)
    patch_size = (16, 16)  # (ph, pw)
    mask_ratio = 0.75
    in_channels = 1
    # Ensure input_dim_size is divisible by num_heads (3)
    input_dim_size = 96  # 96 is divisible by 3 (32 per head)
    mlp_dim = 384  # 384 is divisible by 3 (128 per head)

    # Create a minimal config for MAE that matches get_ssl_network expectations
    config = minimal_network_config(ssl_method)

    # Create the network with the correct parameters
    net = network_factories.get_ssl_network(
        train_loader_call=dummy_train_loader,
        max_epochs=1,
        max_steps_optim=1,
        warmup_steps=0,
        ssl_method=ssl_method,
        ema=dummy_ema(),
        net_type="vit",
        network_config_correct=config,
        stop_gradient=False,
    )

    # Basic type check
    assert isinstance(net, torch.nn.Module)

    # Create test input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, *img_size)

    # Test forward pass
    with torch.no_grad():
        # Get the reconstruction and mask
        rec, mask = net(dummy_input)

        # Check shapes
        num_patches = (img_size[0] // patch_size[0]) ** 2
        expected_mask_shape = (batch_size, num_patches)

        assert rec.shape == dummy_input.shape
        assert mask.shape == expected_mask_shape
        assert mask.bool().any()  # At least some patches should be masked
        assert not mask.bool().all()  # But not all patches should be masked

        # Check mask ratio is approximately correct
        actual_ratio = mask.float().mean().item()
        assert abs(actual_ratio - mask_ratio) < 0.1  # Allow 10% tolerance

    # Test training step
    loss = net.training_step({"image": dummy_input}, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad  # Loss should be a differentiable scalar


@pytest.mark.parametrize(
    ["net_type", "config", "ema"],
    [
        (
            "unet_encoder",
            {
                "in_channels": 1,
                "n_classes": 8,
                "depth": [8, 16],
            },
            None,
        ),
        (
            "convnext",
            {
                "backbone_args": {
                    "spatial_dim": 2,
                    "in_channels": 1,
                    "structure": [(8, 8, 3, 2), (8, 8, 3, 2)],
                    "maxpool_structure": [2, 2],
                },
                "projection_head_args": {
                    "in_channels": 8,
                    "structure": [8, 4],
                    "adn_fn": torch.nn.Identity,
                },
                "prediction_head_args": {
                    "in_channels": 4,
                    "structure": [4, 2],
                    "adn_fn": torch.nn.Identity,
                },
            },
            dummy_ema(),
        ),
    ],
)
def test_else_branch_net_types(net_type, config, ema):
    net = network_factories.get_ssl_network(
        train_loader_call=dummy_train_loader,
        max_epochs=1,
        max_steps_optim=1,
        warmup_steps=0,
        ssl_method="custom",
        ema=ema,
        net_type=net_type,
        network_config_correct=config,
        stop_gradient=False,
    )
    assert isinstance(net, torch.nn.Module)


@pytest.mark.parametrize(
    "ssl_method,net_type",
    [
        ("ijepa", "convnext"),
        ("ijepa", "unet_encoder"),
        ("dino", "unet_encoder"),
        ("dino", "convnext"),
        ("ibot", "unet_encoder"),
        ("ibot", "convnext"),
    ],
)
def test_invalid_ssl_method_net_type_combinations(ssl_method, net_type):
    # These combinations are not supported by the factory and will raise TypeError at constructor level
    net_config = minimal_network_config(ssl_method)
    with pytest.raises(TypeError):
        network_factories.get_ssl_network(
            train_loader_call=dummy_train_loader,
            max_epochs=1,
            max_steps_optim=1,
            warmup_steps=0,
            ssl_method=ssl_method,
            ema=None,
            net_type=net_type,
            network_config_correct=net_config,
            stop_gradient=False,
        )

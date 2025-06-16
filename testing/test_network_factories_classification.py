import pytest
import torch
from adell_mri.utils.network_factories import get_classification_network


# Dummy train loader callable
def dummy_loader():
    return None


# Minimal common arguments
COMMON_ARGS = {
    "dropout_param": 0.1,
    "seed": 42,
    "n_classes": 2,
    "keys": ["image"],
    "train_loader_call": dummy_loader,
    "max_epochs": 1,
    "warmup_steps": 0,
    "start_decay": 0,
    "crop_size": (8, 8, 8),  # 3D for factorized_vit, works for others
}


def test_unet():
    network_config = {
        "activation_fn": torch.nn.ReLU,
        "depth": [64, 64],
    }
    args = COMMON_ARGS.copy()
    args.update(
        {
            "net_type": "unet",
            "network_config": network_config,
            "clinical_feature_keys": [],
        }
    )
    get_classification_network(**args)


def test_vit():
    network_config = {
        "act_fn": "relu",
        "norm_fn": "batch",
        "patch_size": (4, 4, 4),
        "number_of_blocks": 1,
        "attention_dim": 8,
        "n_heads": 1,
        "mlp_structure": [8],
    }
    args = COMMON_ARGS.copy()
    args.update(
        {
            "net_type": "vit",
            "network_config": network_config,
            "clinical_feature_keys": [],
        }
    )
    net = get_classification_network(**args)
    assert "ViTClassifierPL" in type(net).__name__
    # Forward pass
    batch_size = 2
    channels = 1
    D, H, W = args["crop_size"]
    x = torch.randn(batch_size, channels, D, H, W)
    out = net(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == batch_size


def test_factorized_vit():
    network_config = {
        "act_fn": "relu",
        "norm_fn": "batch",
        "patch_size": (4, 4, 4),
        "number_of_blocks": 1,
        "attention_dim": 8,
        "n_heads": 1,
        "mlp_structure": [8],
    }
    args = COMMON_ARGS.copy()
    args.update(
        {
            "net_type": "factorized_vit",
            "network_config": network_config,
            "clinical_feature_keys": [],
        }
    )
    net = get_classification_network(**args)
    assert "FactorizedViTClassifierPL" in type(net).__name__
    # Forward pass
    batch_size = 2
    channels = 1
    D, H, W = args["crop_size"]
    x = torch.randn(batch_size, channels, D, H, W)
    out = net(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == batch_size


def test_classnetpl():
    network_config = {
        "act_fn": "relu",
        "norm_fn": "batch",
    }
    args = COMMON_ARGS.copy()
    args.update(
        {
            "net_type": "cat",
            "network_config": network_config,
            "clinical_feature_keys": [],
        }
    )
    net = get_classification_network(**args)
    assert "ClassNetPL" in type(net).__name__
    # Forward pass
    batch_size = 2
    channels = 1
    D, H, W = args["crop_size"]
    x = torch.randn(batch_size, channels, D, H, W)
    out = net(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == batch_size


def test_hybrid_unet():
    network_config = {
        "activation_fn": torch.nn.ReLU,
        "depth": [64, 64],
    }
    args = COMMON_ARGS.copy()
    args.update(
        {
            "net_type": "unet",
            "network_config": network_config,
            "clinical_feature_keys": ["age", "psa"],
            "clinical_feature_means": torch.zeros(2),
            "clinical_feature_stds": torch.ones(2),
        }
    )
    get_classification_network(**args)


def test_hybrid_vit():
    network_config = {
        "act_fn": "relu",
        "norm_fn": "batch",
        "patch_size": (4, 4, 4),
        "number_of_blocks": 1,
        "attention_dim": 8,
        "n_heads": 1,
    }
    args = COMMON_ARGS.copy()
    args.update(
        {
            "net_type": "vit",
            "network_config": network_config,
            "clinical_feature_keys": ["age", "psa"],
            "clinical_feature_means": torch.zeros(2),
            "clinical_feature_stds": torch.ones(2),
        }
    )
    net = get_classification_network(**args)
    assert "HybridClassifierPL" in type(net).__name__
    batch_size = 2
    channels = 1
    D, H, W = args["crop_size"]
    n_features = len(args["clinical_feature_keys"])
    x = torch.randn(batch_size, channels, D, H, W)
    x_tab = torch.randn(batch_size, n_features)
    out = net(x, x_tab)
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == batch_size


def test_hybrid_factorized_vit():
    network_config = {
        "act_fn": "relu",
        "norm_fn": "batch",
        "patch_size": (4, 4, 4),
        "number_of_blocks": 1,
        "attention_dim": 8,
        "n_heads": 1,
    }
    args = COMMON_ARGS.copy()
    args.update(
        {
            "net_type": "factorized_vit",
            "network_config": network_config,
            "clinical_feature_keys": ["age", "psa"],
            "clinical_feature_means": torch.zeros(2),
            "clinical_feature_stds": torch.ones(2),
        }
    )
    net = get_classification_network(**args)
    assert "HybridClassifierPL" in type(net).__name__
    batch_size = 2
    channels = 1
    D, H, W = args["crop_size"]
    n_features = len(args["clinical_feature_keys"])
    x = torch.randn(batch_size, channels, D, H, W)
    x_tab = torch.randn(batch_size, n_features)
    out = net(x, x_tab)
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == batch_size


def test_hybrid_classnetpl():
    network_config = {
        "act_fn": "relu",
        "norm_fn": "batch",
    }
    args = COMMON_ARGS.copy()
    args.update(
        {
            "net_type": "cat",
            "network_config": network_config,
            "clinical_feature_keys": ["age", "psa"],
            "clinical_feature_means": torch.zeros(2),
            "clinical_feature_stds": torch.ones(2),
        }
    )
    net = get_classification_network(**args)
    assert "HybridClassifierPL" in type(net).__name__
    batch_size = 2
    channels = 1
    D, H, W = args["crop_size"]
    n_features = len(args["clinical_feature_keys"])
    x = torch.randn(batch_size, channels, D, H, W)
    x_tab = torch.randn(batch_size, n_features)
    out = net(x, x_tab)
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == batch_size

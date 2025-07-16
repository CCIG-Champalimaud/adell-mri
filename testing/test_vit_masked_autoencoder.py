import pytest
import torch
from adell_mri.modules.self_supervised.autoencoders import ViTMaskedAutoEncoder

# Fixtures
@pytest.fixture
def model_config():
    return {
        "image_size": (32, 32),
        "patch_size": (4, 4),
        "in_channels": 3,
        "input_dim_size": 128,
        "encoder_args": {
            "number_of_blocks": 2,
            "n_heads": 4,
            "hidden_dim": 128,
            "mlp_structure": [256],
            "dropout_rate": 0.1,
        },
        "decoder_args": {
            "number_of_blocks": 2,
            "n_heads": 4,
            "hidden_dim": 128,
            "mlp_structure": [256],
            "dropout_rate": 0.1,
        },
    }

@pytest.fixture
def create_model(model_config):
    def _create_model(mask_fraction=0.3, seed=42):
        return ViTMaskedAutoEncoder(
            image_size=model_config["image_size"],
            patch_size=model_config["patch_size"],
            in_channels=model_config["in_channels"],
            input_dim_size=model_config["input_dim_size"],
            encoder_args=model_config["encoder_args"],
            decoder_args=model_config["decoder_args"],
            mask_fraction=mask_fraction,
            seed=seed,
        )
    return _create_model

# Tests
def test_forward_pass(create_model, model_config):
    """
    Test the forward pass of ViTMaskedAutoEncoder.
    """
    model = create_model()
    batch_size = 2

    # Create test input
    x = torch.randn(
        batch_size, model_config["in_channels"], *model_config["image_size"]
    )

    # Forward pass
    output, mask = model(x)

    # Check output shape matches input shape
    assert output.shape == x.shape

    # Check mask shape
    h, w = model_config["image_size"]
    ph, pw = model_config["patch_size"]
    n_patches = (h // ph) * (w // pw)
    assert mask.shape == (batch_size, n_patches)

    # Check mask values are binary
    assert torch.all((mask == 0) | (mask == 1))

def test_mask_fraction(create_model, model_config):
    """
    Test that the mask fraction is approximately correct.
    """
    mask_fraction = 0.4
    model = create_model(mask_fraction=mask_fraction)

    x = torch.randn(1, model_config["in_channels"], *model_config["image_size"])
    _, mask = model(x)

    # Calculate actual mask fraction
    h, w = model_config["image_size"]
    ph, pw = model_config["patch_size"]
    n_patches = (h // ph) * (w // pw)
    actual_mask_fraction = mask.sum().item() / n_patches

    # Allow some tolerance for randomness
    tolerance = 0.1
    assert abs(actual_mask_fraction - mask_fraction) < tolerance, \
        f"Expected mask fraction {mask_fraction}, got {actual_mask_fraction}"

def test_deterministic_masking(create_model, model_config):
    """
Test that masking is deterministic with the same seed."""
    seed = 42
    model1 = create_model(seed=seed)
    model2 = create_model(seed=seed)

    x = torch.randn(1, model_config["in_channels"], *model_config["image_size"])

    # Get masks from both models
    _, mask1 = model1(x)
    _, mask2 = model2(x)

    # Check masks are identical
    assert torch.equal(mask1, mask2)

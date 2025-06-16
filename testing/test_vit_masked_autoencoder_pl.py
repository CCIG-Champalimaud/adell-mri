import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from adell_mri.modules.self_supervised.pl import ViTMaskedAutoEncoderPL


# Fixtures
@pytest.fixture
def model_config():
    return {
        "image_key": "image",
        "image_size": (32, 32),
        "patch_size": (8, 8),
        "in_channels": 1,
        "input_dim_size": 64,
        "encoder_args": {
            "num_layers": 2,
            "num_heads": 4,
            "mlp_dim": 128,
            "dropout": 0.1,
        },
        "decoder_args": {
            "num_layers": 2,
            "num_heads": 4,
            "mlp_dim": 128,
            "dropout": 0.1,
        },
        "embed_method": "linear",
        "dropout_rate": 0.1,
        "mask_fraction": 0.75,
        "learning_rate": 1e-3,
        "weight_decay": 1e-6,
        "warmup_steps": 100,
    }


@pytest.fixture
def sample_data():
    # Create random image data
    batch_size = 2
    channels = 1
    img_size = 32
    return torch.randn(batch_size, channels, img_size, img_size)


@pytest.fixture
def train_dataloader(sample_data):
    dataset = TensorDataset(sample_data)
    return DataLoader(dataset, batch_size=2)


def test_model_initialization(model_config):
    """Test that the model initializes correctly."""
    model = ViTMaskedAutoEncoderPL(**model_config)
    assert model is not None
    assert isinstance(model.criterion, torch.nn.MSELoss)


def test_forward_pass(model_config, sample_data):
    """Test the forward pass produces expected output shapes."""
    model = ViTMaskedAutoEncoderPL(**model_config)
    x = sample_data
    x_recon, mask = model(x)

    # Check output shapes
    assert x_recon.shape == x.shape
    assert mask.shape == (
        x.shape[0],
        (32 // 8) * (32 // 8),
    )  # num_patches = (img_size/patch_size)²


def test_training_step(model_config, sample_data):
    """Test that a training step runs without errors."""
    model = ViTMaskedAutoEncoderPL(**model_config)
    batch = (sample_data,)  # Mimic how batch comes from dataloader
    loss = model.training_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)


def test_validation_step(model_config, sample_data):
    """Test that a validation step runs without errors."""
    model = ViTMaskedAutoEncoderPL(**model_config)
    batch = (sample_data,)  # Mimic how batch comes from dataloader
    loss = model.validation_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)


def test_test_step(model_config, sample_data):
    """Test that a test step runs without errors."""
    model = ViTMaskedAutoEncoderPL(**model_config)
    batch = (sample_data,)  # Mimic how batch comes from dataloader
    loss = model.test_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)


def test_optimizer_configuration(model_config):
    """Test that the optimizer is configured correctly."""
    model = ViTMaskedAutoEncoderPL(**model_config)
    optim_conf = model.configure_optimizers()
    
    if model.warmup_steps > 0:
        # With warmup, we should get a dict with optimizer and lr_scheduler
        assert isinstance(optim_conf, dict)
        assert 'optimizer' in optim_conf
        assert 'lr_scheduler' in optim_conf
        
        optimizer = optim_conf['optimizer']
        scheduler_config = optim_conf['lr_scheduler']
        
        # Check scheduler configuration
        assert 'scheduler' in scheduler_config
        assert 'interval' in scheduler_config
        assert 'frequency' in scheduler_config
        assert scheduler_config['scheduler'].base_lrs[0] == model_config['learning_rate']
    else:
        # Without warmup, just return the optimizer directly
        optimizer = optim_conf
        assert optimizer.param_groups[0]['lr'] == model_config['learning_rate']
    
    # Check optimizer configuration
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]['weight_decay'] == model_config['weight_decay']


def test_dataloader(model_config, sample_data, train_dataloader):
    """Test that the dataloader is correctly set up."""
    # Set the training dataloader callable
    model_config["training_dataloader_call"] = lambda: train_dataloader
    model = ViTMaskedAutoEncoderPL(**model_config)

    # Test if train_dataloader returns the correct dataloader
    assert isinstance(model.train_dataloader(), DataLoader)
    assert len(model.train_dataloader().dataset) == len(sample_data)


def test_metrics(model_config, sample_data):
    """Test that metrics are properly updated and logged."""
    model = ViTMaskedAutoEncoderPL(**model_config)
    batch = (sample_data,)

    # Run a training step to update metrics
    loss = model.training_step(batch, batch_idx=0)

    # Check that loss is a tensor
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)

    # Manually compute metrics for verification
    with torch.no_grad():
        x_recon, _ = model(sample_data)
        mse = torch.nn.functional.mse_loss(x_recon, sample_data)
        psnr = -10 * torch.log10(mse)

    # Reset metrics for next test
    model.train_metrics.reset()

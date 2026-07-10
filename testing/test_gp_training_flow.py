"""
Test the complete GP training flow to verify integration works correctly.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adell_mri.modules.layers.gaussian_process import GaussianProcessLayer


class SimpleGPModel(nn.Module):
    """Simple model with GP head for testing"""

    def __init__(self, input_dim, hidden_dim, output_dim, use_gp=True):
        super().__init__()
        self.use_gp = use_gp
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if use_gp:
            self.prediction_head = nn.Linear(hidden_dim, output_dim)
            self.gaussian_process_head = GaussianProcessLayer(
                output_dim, output_dim
            )
        else:
            self.prediction_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.prediction_head(features)
        if self.use_gp:
            output = self.gaussian_process_head(output)
        return output

    def fit_gp(self, dataloader):
        """Simulate the GP fitting process from pl.py"""
        if self.use_gp:
            for batch in dataloader:
                x, y = batch
                features = self.feature_extractor(x)
                features = self.prediction_head(features)
                self.gaussian_process_head.update_inv_cov(features, y)
            self.gaussian_process_head.get_cov()


def test_gp_training_flow():
    """Test complete GP training flow"""
    input_dim, hidden_dim, output_dim = 64, 32, 8
    batch_size, n_batches = 8, 3

    # Create model with GP
    model = SimpleGPModel(input_dim, hidden_dim, output_dim, use_gp=True)

    # Create synthetic data
    data = []
    for _ in range(n_batches):
        x = torch.randn(batch_size, input_dim)
        # Binary labels
        y = torch.randint(0, 2, [batch_size]).float()
        data.append((x, y))

    # Test forward pass
    x_test = torch.randn(batch_size, input_dim)
    output = model(x_test)
    assert output.shape == (batch_size, output_dim)

    # Test GP fitting
    model.fit_gp(data)

    # Verify covariance matrix was computed
    assert hasattr(model.gaussian_process_head, "cov")
    assert model.gaussian_process_head.cov.shape == (1, output_dim, output_dim)

    # Test sampling after fitting
    features = model.feature_extractor(x_test)
    features = model.prediction_head(features)
    samples = model.gaussian_process_head.rsample(features, n_samples=5)
    assert samples.shape == (5, batch_size, output_dim)

    print("✓ GP training flow test passed")


def test_multiclass_gp_training():
    """Test GP training with multiclass"""
    input_dim, hidden_dim, n_classes = 64, 32, 8
    batch_size, n_batches = 8, 3

    # Create model with GP for multiclass
    model = SimpleGPModel(input_dim, hidden_dim, n_classes, use_gp=True)

    # Create synthetic multiclass data
    data = []
    for _ in range(n_batches):
        x = torch.randn(batch_size, input_dim)
        # One-hot encoded multiclass labels
        y = F.one_hot(
            torch.randint(0, n_classes, [batch_size]), n_classes
        ).float()
        data.append((x, y))

    # Test forward pass
    x_test = torch.randn(batch_size, input_dim)
    output = model(x_test)
    assert output.shape == (batch_size, n_classes)

    # Test GP fitting
    model.fit_gp(data)

    # Verify covariance matrix
    assert model.gaussian_process_head.cov.shape == (1, n_classes, n_classes)

    print("✓ Multiclass GP training test passed")


def test_gp_vs_non_gp_comparison():
    """Test that GP model behaves differently from non-GP model"""
    input_dim, hidden_dim, output_dim = 64, 32, 8
    batch_size = 8

    # Create models
    gp_model = SimpleGPModel(input_dim, hidden_dim, output_dim, use_gp=True)
    non_gp_model = SimpleGPModel(
        input_dim, hidden_dim, output_dim, use_gp=False
    )

    # Same input
    x = torch.randn(batch_size, input_dim)

    # Forward passes
    gp_output = gp_model(x)
    non_gp_output = non_gp_model(x)

    # Outputs should be different (GP adds uncertainty)
    assert not torch.allclose(gp_output, non_gp_output, atol=1e-6)

    print("✓ GP vs non-GP comparison test passed")


def test_gp_edge_cases():
    """Test GP with edge cases"""
    input_dim, output_dim = 32, 2
    batch_size = 4

    gp = GaussianProcessLayer(input_dim, output_dim)

    # Test with single batch
    x_single = torch.randn(1, input_dim)
    y_single = torch.tensor([1.0]).float()

    output = gp(x_single)
    gp.update_inv_cov(x_single, y_single)
    gp.get_cov()

    assert output.shape == (1, output_dim)
    assert gp.cov.shape == (1, output_dim, output_dim)

    # Test with all same labels (edge case for numerical stability)
    x_same = torch.randn(batch_size, input_dim)
    y_same = torch.ones(batch_size).float()

    gp.update_inv_cov(x_same, y_same)
    gp.get_cov()  # Should not crash

    print("✓ GP edge cases test passed")


if __name__ == "__main__":
    test_gp_training_flow()
    test_multiclass_gp_training()
    test_gp_vs_non_gp_comparison()
    test_gp_edge_cases()
    print("\n🎉 All GP training flow tests passed!")

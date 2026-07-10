"""
Test the dedicated predict_step_gp implementation.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from adell_mri.modules.classification.pl import ClassPLABC
from adell_mri.modules.layers.gaussian_process import GaussianProcessLayer


class SimpleGPNetwork(nn.Module):
    """Simple network with GP head for testing"""

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

    def forward_features(self, x):
        """Return features before GP layer"""
        features = self.feature_extractor(x)
        return self.prediction_head(features)


class TestGPPLModule(ClassPLABC):
    """Test Lightning module with GP support"""

    def __init__(self, network, image_key="image", label_key="label"):
        super().__init__()
        self.network = network
        self.image_key = image_key
        self.label_key = label_key
        self.gaussian_process = network.use_gp
        self.n_classes = 2  # Binary classification

        if self.gaussian_process:
            self.gaussian_process_head = network.gaussian_process_head

        self.forward = network.forward

    def training_dataloader_call(self):
        """Mock training dataloader for GP fitting"""
        return self._train_dataloader


def test_dedicated_gp_predict_step():
    """Test the dedicated predict_step_gp method"""
    print("Testing dedicated predict_step_gp method...")

    # Setup
    input_dim, hidden_dim, output_dim = 64, 32, 1
    batch_size = 8
    n_samples = 50

    # Create model with GP
    network = SimpleGPNetwork(input_dim, hidden_dim, output_dim, use_gp=True)
    pl_module = TestGPPLModule(network)

    # Create synthetic data
    x = torch.randn(batch_size, input_dim)
    y = torch.randint(0, 2, [batch_size]).float()
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    pl_module._train_dataloader = dataloader

    # Create batch for prediction
    batch = {pl_module.image_key: x, pl_module.label_key: y}

    # Test 1: Standard predict_step still works
    print("  Test 1: Standard predict_step unchanged...")
    prediction = pl_module.predict_step(batch, 0)
    assert prediction.shape == (batch_size, output_dim)
    print("    ✓ Standard predict_step works unchanged")

    # Test 2: GP fitting
    print("  Test 2: Fitting GP...")
    for batch_train in dataloader:
        x_train, y_train = batch_train
        features = network.forward_features(x_train)
        pl_module.gaussian_process_head.update_inv_cov(features, y_train)
    pl_module.gaussian_process_head.get_cov()
    assert hasattr(pl_module.gaussian_process_head, "cov")
    print("    ✓ GP fitting completed")

    # Test 3: Dedicated GP predict_step
    print("  Test 3: Dedicated predict_step_gp...")
    result = pl_module.predict_step_gp(batch, 0, n_samples=n_samples)

    # Check result structure
    assert isinstance(result, dict), "Result should be a dictionary"
    required_keys = [
        "predictions",
        "prediction_probs",
        "gp_mean",
        "gp_samples",
        "predictive_mean",
        "predictive_std",
        "predictive_variance",
        "predictive_probs",
        "epistemic_uncertainty",
        "total_uncertainty",
    ]

    for key in required_keys:
        assert key in result, f"Missing key: {key}"

    # Check tensor shapes
    assert result["predictions"].shape == (batch_size, output_dim)
    assert result["gp_samples"].shape == (n_samples, batch_size, output_dim)
    assert result["predictive_mean"].shape == (batch_size, output_dim)
    assert result["predictive_std"].shape == (batch_size, output_dim)
    assert result["epistemic_uncertainty"].shape == (batch_size, output_dim)

    print("    ✓ Dedicated GP predict_step works")

    # Test 4: Non-GP model should raise error
    print("  Test 4: Non-GP model should raise error...")
    network_no_gp = SimpleGPNetwork(
        input_dim, hidden_dim, output_dim, use_gp=False
    )
    pl_module_no_gp = TestGPPLModule(network_no_gp)

    try:
        pl_module_no_gp.predict_step_gp(batch, 0)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "Gaussian process is not enabled" in str(e)
        print("    ✓ Non-GP model correctly raises error")

    # Test 5: GP not fitted should raise error
    print("  Test 5: GP not fitted should raise error...")
    network_unfitted = SimpleGPNetwork(
        input_dim, hidden_dim, output_dim, use_gp=True
    )
    pl_module_unfitted = TestGPPLModule(network_unfitted)

    try:
        pl_module_unfitted.predict_step_gp(batch, 0)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not fitted" in str(e)
        print("    ✓ Unfitted GP correctly raises error")

    print("\n🎉 All dedicated GP predict_step tests passed!")


def test_multiclass_dedicated_gp():
    """Test dedicated GP predict_step with multiclass"""
    print("Testing multiclass dedicated GP predict_step...")

    # Setup
    input_dim, hidden_dim, n_classes = 64, 32, 4
    batch_size = 8
    n_samples = 30

    # Create multiclass model with GP
    network = SimpleGPNetwork(input_dim, hidden_dim, n_classes, use_gp=True)
    pl_module = TestGPPLModule(network)
    pl_module.n_classes = n_classes

    # Create synthetic multiclass data
    x = torch.randn(batch_size, input_dim)
    y = F.one_hot(torch.randint(0, n_classes, [batch_size]), n_classes).float()
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    pl_module._train_dataloader = dataloader

    # Fit GP
    for batch_train in dataloader:
        x_train, y_train = batch_train
        features = network.forward_features(x_train)
        pl_module.gaussian_process_head.update_inv_cov(features, y_train)
    pl_module.gaussian_process_head.get_cov()

    # Test prediction with uncertainty
    batch = {pl_module.image_key: x, pl_module.label_key: y}
    result = pl_module.predict_step_gp(batch, 0, n_samples=n_samples)

    # Check multiclass specific shapes and properties
    assert result["predictions"].shape == (batch_size, n_classes)
    assert result["predictive_probs"].shape == (batch_size, n_classes)

    # Check softmax properties (sums to 1)
    prob_sums = result["predictive_probs"].sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6)

    print("  ✓ Multiclass dedicated GP prediction works")
    print("🎉 Multiclass test passed!")


if __name__ == "__main__":
    test_dedicated_gp_predict_step()
    test_multiclass_dedicated_gp()
    print("\n✅ All dedicated GP predict_step tests completed successfully!")

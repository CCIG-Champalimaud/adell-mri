import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pytest
import torch

from adell_mri.modules.segmentation.losses import (
    binary_cross_entropy,
    binary_focal_loss,
    binary_generalized_dice_loss,
    cat_cross_entropy,
    combo_loss,
    generalised_dice_score,
    mc_focal_loss,
    mc_generalized_dice_loss,
    unified_focal_loss,
)

EPS = 1e-6

BINARY_PRED = torch.tensor([[[[0.9, 0.1], [0.4, 0.6]]]], dtype=torch.float64)
BINARY_TARGET = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float64)

MC_LOGITS = torch.tensor(
    [
        [[2.0, 0.5], [0.3, 1.5], [0.1, 2.5]],
        [[1.0, 0.4], [0.3, 0.2], [2.5, 0.1]],
    ],
    dtype=torch.float64,
)
MC_PRED = torch.softmax(MC_LOGITS, dim=1)
MC_TARGET = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)


def _one_hot(target, n_classes=3):
    """Replicates classes_to_one_hot for a [batch, spatial] target."""
    return (
        torch.nn.functional.one_hot(target.long(), num_classes=n_classes)
        .permute(0, 2, 1)
        .to(torch.float64)
    )


def test_generalised_dice_score_perfect_match_is_half():
    score = generalised_dice_score(BINARY_TARGET, BINARY_TARGET)
    assert torch.allclose(score, torch.tensor(0.5, dtype=torch.float64))


def test_binary_cross_entropy_value():
    loss = binary_cross_entropy(BINARY_PRED, BINARY_TARGET)
    expected = -torch.mean(
        BINARY_TARGET * torch.log(BINARY_PRED + EPS)
        + (1 - BINARY_TARGET) * torch.log(1 - BINARY_PRED + EPS)
    )
    assert torch.allclose(loss, expected)


def test_binary_focal_loss_value():
    """
    Focal loss per Lin et al. (2018): -(1-pt)^gamma * log(pt), with the
    repository's convention of applying ``alpha`` to positive-class terms.
    """
    gamma = 2.0
    loss = binary_focal_loss(BINARY_PRED, BINARY_TARGET, gamma=gamma)
    p = BINARY_PRED.flatten(start_dim=2)
    t = (BINARY_TARGET > 0.5).long().flatten(start_dim=2).to(torch.float64)
    pt = t * p + (1 - t) * (1 - p)
    expected = -torch.mean(t * (1 - pt) ** gamma * torch.log(pt), dim=-1)
    assert loss.shape == expected.shape
    assert torch.allclose(loss, expected)


def test_focal_loss_downweights_easy_examples():
    """
    The focusing parameter must down-weight easy (pt close to 1) examples:
    loss(gamma>0) < BCE for confident correct predictions.
    """
    pred = torch.tensor([[[[0.99]]]], dtype=torch.float64)
    target = torch.tensor([[[[1.0]]]], dtype=torch.float64)
    bce = binary_cross_entropy(pred, target)
    focal = binary_focal_loss(pred, target, gamma=2.0)
    assert float(focal) < float(bce)


def test_cat_cross_entropy_value_and_smoothing():
    """
    Label smoothing must interpolate targets toward 1/n_classes and be a
    no-op when label_smoothing=0.
    """
    one_hot = _one_hot(MC_TARGET)
    n_classes = one_hot.shape[1]

    loss = cat_cross_entropy(MC_PRED, MC_TARGET)
    expected = -torch.mean(
        torch.flatten(one_hot * torch.log(MC_PRED + EPS), start_dim=1), dim=1
    )
    assert torch.allclose(loss, expected)

    ls = 0.3
    smoothed_loss = cat_cross_entropy(MC_PRED, MC_TARGET, label_smoothing=ls)
    shifted_target = one_hot * (1 - ls) + ls / n_classes
    expected_smoothed = -torch.mean(
        torch.flatten(shifted_target * torch.log(MC_PRED + EPS), start_dim=1),
        dim=1,
    )
    assert torch.allclose(smoothed_loss, expected_smoothed)


def test_mc_focal_loss_value():
    gamma = 2.0
    alpha = torch.ones(3, dtype=torch.float64)
    loss = mc_focal_loss(MC_PRED, MC_TARGET, alpha=alpha, gamma=gamma)
    one_hot = _one_hot(MC_TARGET)
    pt = torch.where(one_hot > 0.5, MC_PRED, 1 - MC_PRED)
    ce = -one_hot * torch.log(MC_PRED + EPS)
    expected = torch.mean(
        torch.flatten((1 - pt + EPS) ** gamma * ce, start_dim=1), dim=1
    )
    assert loss.shape == expected.shape
    assert torch.allclose(loss, expected)


def test_binary_generalized_dice_loss_value_with_default_smoothing():
    loss = binary_generalized_dice_loss(BINARY_TARGET, BINARY_TARGET)
    s = BINARY_TARGET.sum()
    n = BINARY_TARGET.numel()
    gds = s / (2 * s + n)
    assert torch.allclose(loss, torch.tensor(1 - 2 * gds, dtype=torch.float64))


def test_mc_generalized_dice_loss_value():
    one_hot = _one_hot(MC_TARGET)
    loss = mc_generalized_dice_loss(MC_PRED, one_hot)
    assert loss.shape == (MC_TARGET.shape[0],)
    assert bool((loss > 0.0).all())


def test_combo_loss_value():
    loss = combo_loss(BINARY_PRED, BINARY_TARGET)
    assert float(loss) > 0.0


def test_unified_focal_loss_value():
    loss = unified_focal_loss(BINARY_PRED, BINARY_TARGET, weight=0.5, gamma=0.5)
    assert float(loss) > 0.0


def test_losses_deterministic():
    a = binary_focal_loss(BINARY_PRED, BINARY_TARGET, gamma=2.0)
    b = binary_focal_loss(BINARY_PRED, BINARY_TARGET, gamma=2.0)
    assert torch.equal(a, b)


@pytest.mark.parametrize("fn", [combo_loss, unified_focal_loss])
def test_composite_losses_deterministic(fn):
    kwargs = {"weight": 0.5, "gamma": 0.5} if fn is unified_focal_loss else {}
    a = fn(BINARY_PRED, BINARY_TARGET, **kwargs)
    b = fn(BINARY_PRED, BINARY_TARGET, **kwargs)
    assert torch.equal(a, b)


def test_barlow_twins_loss_multiple_updates_non_moving():
    """
    Non-moving BarlowTwinsLoss must standardize per batch and support
    repeated update=True calls (previously crashed on the second call).
    """
    from adell_mri.modules.self_supervised.losses.barlow_twins import (
        BarlowTwinsLoss,
    )

    torch.manual_seed(0)
    loss_fn = BarlowTwinsLoss(moving=False)
    for _ in range(3):
        x, y = torch.rand(4, 8), torch.rand(4, 8)
        loss = loss_fn(x, y, update=True)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


def test_barlow_twins_loss_identical_views_zero_invariance():
    """
    For identical views the sample-vs-sample cross-correlation diagonal is
    exactly one, so the invariance term must vanish; the remaining value
    comes solely from the scaled off-diagonal (reduction) term.
    """
    from adell_mri.modules.self_supervised.losses.barlow_twins import (
        BarlowTwinsLoss,
    )

    torch.manual_seed(0)
    x = torch.rand(16, 8, dtype=torch.float64)
    loss_fn = BarlowTwinsLoss(moving=False)
    C = loss_fn.pearson_corr(x, x.clone())
    inv_term = torch.diagonal(1 - C).abs().max()
    assert float(inv_term) < 1e-12


def test_barlow_twins_standardize_modes():
    """
    Non-moving mode must always use batch statistics; moving mode must use
    the running statistics once they are available (previously crashed on
    the second update in non-moving mode).
    """
    from adell_mri.modules.self_supervised.losses.barlow_twins import (
        BarlowTwinsLoss,
    )

    torch.manual_seed(0)
    x = torch.rand(32, 8, dtype=torch.float64) * 3 + 5

    fn = BarlowTwinsLoss(moving=False)
    out = fn.standardize(x)
    assert torch.allclose(out.mean(0), torch.zeros(8, dtype=torch.float64))
    assert torch.allclose(out.std(0), torch.ones(8, dtype=torch.float64))

    # moving mode: populate running statistics from one batch...
    fn = BarlowTwinsLoss(moving=True)
    y = x + torch.rand(32, 8, dtype=torch.float64) * 0.01
    loss = fn(x, y, update=True)
    assert torch.isfinite(loss)
    # ...then standardize a batch from a very different distribution: the
    # output must NOT be batch-standardized (running stats are used)
    z = torch.rand(32, 8, dtype=torch.float64) * 3 + 50
    out = fn.standardize(z)
    assert not torch.allclose(out.mean(0), torch.zeros(8, dtype=torch.float64))


def test_barlow_twins_loss_moving_average_mode():
    """Moving-average mode must work across multiple updates."""
    from adell_mri.modules.self_supervised.losses.barlow_twins import (
        BarlowTwinsLoss,
    )

    torch.manual_seed(0)
    loss_fn = BarlowTwinsLoss(moving=True)
    for _ in range(3):
        x, y = torch.rand(4, 8), torch.rand(4, 8)
        loss = loss_fn(x, y, update=True)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

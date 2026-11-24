import torch

from adell_mri.utils.pl_utils import get_batch_size_and_steps_per_epoch
from adell_mri.utils.samplers import get_train_sampler


def test_get_batch_size_and_steps_per_epoch():
    bs, real_bs, steps = get_batch_size_and_steps_per_epoch(16, 1, 3844)
    assert (bs, real_bs, steps) == (16, 16, 240)

    _, _, steps = get_batch_size_and_steps_per_epoch(
        16, 2, 3844, steps_per_epoch=250
    )
    assert steps == 250

    bs, real_bs, steps = get_batch_size_and_steps_per_epoch(16, 2, 8)
    assert (bs, real_bs, steps) == (4, 8, 1)

    _, _, steps = get_batch_size_and_steps_per_epoch(4, 1, 100, n_samples=1000)
    assert steps == 250


def test_get_train_sampler():
    dataset = list(range(100))
    generator = torch.Generator().manual_seed(0)

    sampler = get_train_sampler(
        dataset, steps_per_epoch=10, batch_size=4, generator=generator
    )
    assert len(sampler) == 40
    assert len(list(sampler)) == 40

    sampler = get_train_sampler(
        dataset, steps_per_epoch=100, batch_size=4, generator=generator
    )
    assert len(list(sampler)) == 400

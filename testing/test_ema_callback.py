import torch

from adell_mri.utils.pl_callbacks import EMACallback


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = torch.nn.Linear(2, 2, bias=False)
        self.discriminator = torch.nn.Linear(2, 2, bias=False)


def test_ema_excludes_discriminator():
    model = _Model()
    callback = EMACallback(
        decay=0.5,
        use_ema_weights=False,
        exclude_prefixes=["discriminator"],
    )
    callback.on_fit_start(None, model)
    callback.on_train_batch_end(None, model, None, None, 0)

    initial_discriminator = model.discriminator.weight.clone()
    with torch.no_grad():
        model.generator.weight.add_(2.0)
        model.discriminator.weight.add_(5.0)

    callback.on_train_batch_end(None, model, None, None, 1)

    assert torch.equal(
        callback.ema.shadow.discriminator.weight, initial_discriminator
    )
    assert not torch.equal(
        callback.ema.shadow.discriminator.weight,
        model.discriminator.weight,
    )

    checkpoint = {}
    callback.on_save_checkpoint(None, model, checkpoint)
    assert all(
        "discriminator" not in key for key in checkpoint["state_dict_ema"]
    )


def test_ema_end_does_not_overwrite_discriminator():
    model = _Model()
    callback = EMACallback(
        decay=0.5,
        use_ema_weights=True,
        exclude_prefixes=["discriminator"],
    )
    callback.on_fit_start(None, model)
    callback.on_train_batch_end(None, model, None, None, 0)

    with torch.no_grad():
        model.generator.weight.add_(2.0)
        model.discriminator.weight.add_(5.0)

    callback.on_train_batch_end(None, model, None, None, 1)
    live_discriminator = model.discriminator.weight.clone()

    callback.on_train_end(None, model)

    assert torch.equal(model.discriminator.weight, live_discriminator)

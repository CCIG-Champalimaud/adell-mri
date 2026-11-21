import torch

from adell_mri.utils.pl_callbacks import EMACallback


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = torch.nn.Linear(2, 2, bias=False)
        self.discriminator = torch.nn.Linear(2, 2, bias=False)


def test_ema_validation_before_training_is_noop():
    model = _Model()
    callback = EMACallback(
        decay=0.5,
        use_ema_weights=True,
        exclude_prefixes=["discriminator"],
    )
    callback.on_fit_start(None, model)
    initial_generator = model.generator.weight.clone()
    initial_discriminator = model.discriminator.weight.clone()

    callback.on_validation_epoch_start(None, model)
    callback.on_validation_end(None, model)
    callback.on_train_end(None, model)

    assert torch.equal(model.generator.weight, initial_generator)
    assert torch.equal(model.discriminator.weight, initial_discriminator)


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


class _Trainer:
    def __init__(self, global_step=0):
        self.global_step = global_step


def test_ema_waits_for_start_steps():
    model = _Model()
    callback = EMACallback(
        decay=0.5,
        start_steps=2,
        use_ema_weights=False,
    )
    callback.on_fit_start(None, model)
    trainer = _Trainer(global_step=0)
    initial_generator = model.generator.weight.clone()

    callback.on_train_batch_end(trainer, model, None, None, 0)
    assert callback.ema.shadow is None

    # validation before start_steps must use the raw weights
    callback.on_validation_epoch_start(trainer, model)
    callback.on_validation_end(trainer, model)
    assert torch.equal(model.generator.weight, initial_generator)

    trainer.global_step = 2
    callback.on_train_batch_end(trainer, model, None, None, 2)
    assert callback.ema.shadow is not None

    with torch.no_grad():
        model.generator.weight.add_(2.0)
    trainer.global_step = 3
    callback.on_train_batch_end(trainer, model, None, None, 3)
    assert not torch.equal(
        callback.ema.shadow.generator.weight, model.generator.weight
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

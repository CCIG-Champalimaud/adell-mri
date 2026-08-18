import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lightning.pytorch as pl
import pytest
import torch

from adell_mri.modules.gan.discriminator import Discriminator
from adell_mri.modules.gan.generator import Generator
from adell_mri.modules.gan.losses import (
    AdversarialLoss,
    RelativisticGANLoss,
    SemiSLAdversarialLoss,
    SemiSLRelativisticGANLoss,
    SemiSLWGANGPLoss,
    WGANGPLoss,
    reduce_losses,
)
from adell_mri.modules.gan.pl import GANPL, RelativisticGANPL

context_dim = 32
n_class_embeds = 4


def test_generator_standard():
    input_tensor = torch.rand(1, 1, 32, 32)
    generator = Generator(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
    )
    generator(input_tensor)


def test_generator_with_class_embeddings():
    input_tensor = torch.rand(1, 1, 32, 32)
    generator = Generator(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_class_embeds=n_class_embeds,
    )
    generator(
        input_tensor,
        class_labels=torch.randint(low=0, high=n_class_embeds, size=(1,)),
    )


def test_generator_with_cross_attention():
    input_tensor = torch.rand(1, 1, 32, 32)
    generator = Generator(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        cross_attention_dim=context_dim,
        with_conditioning=True,
    )
    generator(input_tensor, context=torch.rand(1, 1, context_dim))


def test_generator_with_cross_attention_and_class_embeddings():
    input_tensor = torch.rand(1, 1, 32, 32)
    context_dim = 64
    generator = Generator(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_class_embeds=n_class_embeds,
        cross_attention_dim=context_dim,
        with_conditioning=True,
    )
    generator(
        input_tensor,
        context=torch.rand(1, 1, generator.cross_attention_dim),
        class_labels=torch.randint(low=0, high=n_class_embeds, size=(1,)),
    )


@pytest.mark.parametrize("context_dim", [(32,), (64,)])
def test_generator_with_cross_attention_and_class_embeddings(context_dim):
    input_tensor = torch.rand(1, 1, 32, 32)
    context_dim = 32
    generator = Generator(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_class_embeds=n_class_embeds,
        cross_attention_dim=context_dim,
        with_conditioning=True,
    )
    generator(
        input_tensor,
        context=torch.rand(1, 1, generator.cross_attention_dim),
        class_labels=torch.randint(low=0, high=n_class_embeds, size=(1,)),
    )


def test_discriminator_convnext():
    disc = Discriminator(
        "convnext",
        in_channels=1,
        spatial_dim=2,
        structure=[[16, 16, 3, 3], [32, 32, 3, 3]],
    )

    input_tensor = torch.rand(1, 1, 32, 32)
    out, class_target, reg_target, features = disc(input_tensor)
    assert list(out.shape) == [1, 1]


def test_discriminator_convnext_additional_classifiers():
    disc = Discriminator(
        "convnext",
        additional_classification_targets=[2, 4],
        in_channels=1,
        spatial_dim=2,
        structure=[[16, 16, 3, 3], [32, 32, 3, 3]],
    )

    input_tensor = torch.rand(1, 1, 32, 32)
    out, class_target, reg_target, features = disc(input_tensor)
    assert list(out.shape) == [1, 1]
    assert list(class_target[0].shape) == [1, 2]
    assert list(class_target[1].shape) == [1, 4]


def test_discriminator_convnext_additional_regressors():
    disc = Discriminator(
        "convnext",
        additional_regression_targets=4,
        in_channels=1,
        spatial_dim=2,
        structure=[[16, 16, 3, 3], [32, 32, 3, 3]],
    )

    input_tensor = torch.rand(1, 1, 32, 32)
    out, class_target, reg_target, features = disc(input_tensor)
    assert list(out.shape) == [1, 1]
    assert list(reg_target.shape) == [1, 4]


def test_discriminator_convnext_additional_regressors_and_classifiers():
    disc = Discriminator(
        "convnext",
        additional_classification_targets=[2, 4],
        additional_regression_targets=4,
        in_channels=1,
        spatial_dim=2,
        structure=[[16, 16, 3, 3], [32, 32, 3, 3]],
    )

    input_tensor = torch.rand(1, 1, 32, 32)
    out, class_target, reg_target, features = disc(input_tensor)
    assert list(out.shape) == [1, 1]
    assert list(class_target[0].shape) == [1, 2]
    assert list(class_target[1].shape) == [1, 4]
    assert list(reg_target.shape) == [1, 4]


def test_discriminator_convnext_additional_classifiers_and_features():
    disc = Discriminator(
        "convnext",
        additional_classification_targets=[2, 4],
        additional_features=4,
        in_channels=1,
        spatial_dim=2,
        structure=[[16, 16, 3, 3], [32, 32, 3, 3]],
    )

    input_tensor = torch.rand(1, 1, 32, 32)
    additional_input_tensor = torch.rand(1, 4)
    out, class_target, reg_target, features = disc(
        input_tensor, additional_input_tensor
    )
    assert list(out.shape) == [1, 1]
    assert list(class_target[0].shape) == [1, 2]
    assert list(class_target[1].shape) == [1, 4]


def test_gan_complete():
    generator = Generator(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        cross_attention_dim=context_dim,
        with_conditioning=True,
    )
    disc = Discriminator(
        "convnext",
        additional_classification_targets=[2, 4],
        in_channels=1,
        spatial_dim=2,
        structure=[[16, 16, 3, 3], [32, 32, 3, 3]],
    )

    gan = GANPL(
        generator=generator,
        discriminator=disc,
        classification_target_key="class",
        class_target_specification=[n_class_embeds],
    )

    cl = torch.randint(low=0, high=n_class_embeds, size=(1, 1))

    input_tensor = torch.rand(1, 1, 32, 32)
    gen_output, _, _ = gan(input_tensor, class_target=cl)

    gan.discriminator(gen_output)


def make_gan_pair(n_dim=2):
    generator = Generator(
        spatial_dims=n_dim,
        in_channels=1,
        out_channels=1,
        num_channels=[32, 32],
        num_res_blocks=1,
        attention_levels=(False, False),
    )
    disc = Discriminator(
        "convnext",
        in_channels=1,
        spatial_dim=n_dim,
        structure=[[16, 16, 3, 3], [32, 32, 3, 3]],
    )
    return generator, disc


def test_reduce_losses_scalar_and_vector():
    scalar = torch.tensor(1.0)
    vector = torch.rand(4)
    loss = reduce_losses({"scalar": scalar, "vector": vector})
    assert loss.ndim == 0
    assert torch.allclose(loss, torch.as_tensor(1.0 + vector.mean()))


def test_adversarial_loss_discriminator_direction():
    generator, disc = make_gan_pair()
    gen_samples = torch.rand(2, 1, 16, 16)
    real_samples = torch.rand(2, 1, 16, 16)
    loss = AdversarialLoss()
    # use logits directly through discriminator
    gen_pred = disc(gen_samples)[0]
    real_pred = disc(real_samples)[0]
    gen_loss = loss.generator_loss(gen_pred)["adversarial"]
    disc_loss = loss.discriminator_loss(gen_pred=gen_pred, real_pred=real_pred)[
        "adversarial"
    ]
    assert gen_loss.ndim == 0
    assert disc_loss.ndim == 0
    total = loss(gen_samples, real_samples, disc)
    assert total.ndim == 0


def test_wgan_gp_loss_has_gradient_penalty():
    generator, disc = make_gan_pair()
    disc.train()
    gen_samples = torch.rand(2, 1, 16, 16)
    real_samples = torch.rand(2, 1, 16, 16)
    loss = WGANGPLoss(lambda_gp=10.0)
    losses = loss.discriminator_loss(
        gen_samples=gen_samples,
        real_samples=real_samples,
        discriminator=disc,
        gen_pred=disc(gen_samples)[0],
        real_pred=disc(real_samples)[0],
    )
    assert "gradient_penalty" in losses
    total = loss(gen_samples, real_samples, disc)
    assert total.ndim == 0


def test_relativistic_gan_loss():
    generator, disc = make_gan_pair()
    disc.train()
    gen_samples = torch.rand(2, 1, 16, 16)
    real_samples = torch.rand(2, 1, 16, 16)
    loss = RelativisticGANLoss(lambda_gp=1.0)
    total = loss(gen_samples, real_samples, disc)
    assert total.ndim == 0
    avg_loss = RelativisticGANLoss(lambda_gp=1.0, average=True)
    total_avg = avg_loss(gen_samples, real_samples, disc)
    assert total_avg.ndim == 0


def test_semisupervised_gan_losses_class_and_reg():
    disc = Discriminator(
        "convnext",
        additional_classification_targets=[2],
        additional_regression_targets=1,
        in_channels=1,
        spatial_dim=2,
        structure=[[16, 16, 3, 3], [32, 32, 3, 3]],
    )
    disc.train()
    gen_samples = torch.rand(2, 1, 16, 16)
    real_samples = torch.rand(2, 1, 16, 16)
    gen_pred, gen_class_pred, gen_reg_pred, _ = disc(gen_samples)
    real_pred, real_class_pred, real_reg_pred, _ = disc(real_samples)
    class_target = [torch.randint(0, 2, size=(2,))]
    reg_target = [torch.rand(2, 1)]
    for loss in [
        SemiSLAdversarialLoss(),
        SemiSLWGANGPLoss(lambda_gp=10.0),
    ]:
        gen_losses = loss.generator_loss(
            gen_pred=gen_pred,
            class_pred=gen_class_pred,
            class_target=class_target,
            reg_pred=gen_reg_pred,
            reg_target=reg_target,
        )
        assert {"adversarial", "class", "reg"} <= set(gen_losses.keys())
        disc_losses = loss.discriminator_loss(
            gen_pred=gen_pred,
            real_pred=real_pred,
            gen_samples=gen_samples,
            real_samples=real_samples,
            discriminator=disc,
            gen_class_pred=gen_class_pred,
            real_class_pred=real_class_pred,
            class_target=class_target,
            gen_reg_pred=gen_reg_pred,
            real_reg_pred=real_reg_pred,
            reg_target=reg_target,
        )
        assert {"adversarial", "class", "reg"} <= set(disc_losses.keys())
    rel_loss = SemiSLRelativisticGANLoss(lambda_gp=1.0)
    gen_losses = rel_loss.generator_loss(
        gen_pred=gen_pred,
        real_pred=real_pred,
        class_pred=gen_class_pred,
        class_target=class_target,
        reg_pred=gen_reg_pred,
        reg_target=reg_target,
    )
    assert {"adversarial", "class", "reg"} <= set(gen_losses.keys())
    disc_losses = rel_loss.discriminator_loss(
        gen_pred=gen_pred,
        real_pred=real_pred,
        gen_samples=gen_samples,
        real_samples=real_samples,
        discriminator=disc,
        gen_class_pred=gen_class_pred,
        real_class_pred=real_class_pred,
        class_target=class_target,
        gen_reg_pred=gen_reg_pred,
        real_reg_pred=real_reg_pred,
        reg_target=reg_target,
    )
    assert {"adversarial", "class", "reg"} <= set(disc_losses.keys())


@pytest.mark.parametrize("n_dim", [2, 3])
def test_ganpl_training_loop(n_dim):
    size = [1, 16, 16] if n_dim == 2 else [1, 16, 16, 16]
    generator = Generator(
        spatial_dims=n_dim,
        in_channels=1,
        out_channels=1,
        num_channels=[32, 32],
        num_res_blocks=1,
        attention_levels=(False, False),
    )
    disc = Discriminator(
        "convnext",
        in_channels=1,
        spatial_dim=n_dim,
        structure=[[16, 16, 3, 3], [32, 32, 3, 3]],
    )

    def dl(batch_size):
        return torch.utils.data.DataLoader(
            [{"real_image": torch.rand(*size)} for _ in range(4)],
            batch_size=batch_size,
        )

    gan = GANPL(
        generator=generator,
        discriminator=disc,
        training_dataloader_call=dl,
        batch_size=2,
    )

    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        accelerator="cpu",
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(gan)
    assert gan.adversarial_loss is not None


@pytest.mark.parametrize("n_dim", [2, 3])
def test_ganpl_training_loop_wgan(n_dim):
    size = [1, 16, 16] if n_dim == 2 else [1, 16, 16, 16]
    generator = Generator(
        spatial_dims=n_dim,
        in_channels=1,
        out_channels=1,
        num_channels=[32, 32],
        num_res_blocks=1,
        attention_levels=(False, False),
    )
    disc = Discriminator(
        "convnext",
        in_channels=1,
        spatial_dim=n_dim,
        structure=[[16, 16, 3, 3], [32, 32, 3, 3]],
    )

    def dl(batch_size):
        return torch.utils.data.DataLoader(
            [{"real_image": torch.rand(*size)} for _ in range(4)],
            batch_size=batch_size,
        )

    gan = GANPL(
        generator=generator,
        discriminator=disc,
        lambda_gp=10.0,
        training_dataloader_call=dl,
        batch_size=2,
    )

    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        accelerator="cpu",
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(gan)
    assert isinstance(gan.adversarial_loss, SemiSLWGANGPLoss)


@pytest.mark.parametrize("n_dim", [2, 3])
def test_relativistic_ganpl_training_loop(n_dim):
    size = [1, 16, 16] if n_dim == 2 else [1, 16, 16, 16]
    generator = Generator(
        spatial_dims=n_dim,
        in_channels=1,
        out_channels=1,
        num_channels=[32, 32],
        num_res_blocks=1,
        attention_levels=(False, False),
    )
    disc = Discriminator(
        "convnext",
        in_channels=1,
        spatial_dim=n_dim,
        structure=[[16, 16, 3, 3], [32, 32, 3, 3]],
    )

    def dl(batch_size):
        return torch.utils.data.DataLoader(
            [{"real_image": torch.rand(*size)} for _ in range(4)],
            batch_size=batch_size,
        )

    gan = RelativisticGANPL(
        generator=generator,
        discriminator=disc,
        lambda_gp=1.0,
        training_dataloader_call=dl,
        batch_size=2,
    )

    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        accelerator="cpu",
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(gan)
    assert isinstance(gan.adversarial_loss, SemiSLRelativisticGANLoss)


def test_ganpl_conditional_training_loop():
    size = [1, 16, 16]
    generator = Generator(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=[32, 32],
        num_res_blocks=1,
        attention_levels=(False, False),
        cross_attention_dim=32,
        with_conditioning=True,
    )
    disc = Discriminator(
        "convnext",
        in_channels=2,
        spatial_dim=2,
        structure=[[16, 16, 3, 3], [32, 32, 3, 3]],
    )

    def dl(batch_size):
        return torch.utils.data.DataLoader(
            [
                {"real_image": torch.rand(*size), "input": torch.rand(*size)}
                for _ in range(4)
            ],
            batch_size=batch_size,
        )

    gan = GANPL(
        generator=generator,
        discriminator=disc,
        input_image_key="input",
        training_dataloader_call=dl,
        batch_size=2,
    )

    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        accelerator="cpu",
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(gan)
    assert gan.input_image_key == "input"

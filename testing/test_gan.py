import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pytest

import torch
from adell_mri.modules.gan.discriminator import Discriminator
from adell_mri.modules.gan.generator import Generator
from adell_mri.modules.gan.gan import GAN

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
    out, class_target, reg_target = disc(input_tensor)
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
    out, class_target, reg_target = disc(input_tensor)
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
    out, class_target, reg_target = disc(input_tensor)
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
    out, class_target, reg_target = disc(input_tensor)
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
    out, class_target, reg_target = disc(input_tensor, additional_input_tensor)
    assert list(out.shape) == [1, 1]
    assert list(class_target[0].shape) == [1, 2]
    assert list(class_target[1].shape) == [1, 4]


def test_gan_complete():
    generator = Generator(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_class_embeds=n_class_embeds,
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

    gan = GAN(generator=generator, discriminator=disc)

    cl = torch.randint(low=0, high=n_class_embeds, size=(1,))

    input_tensor = torch.rand(1, 1, 32, 32)
    gen_output = gan(
        input_tensor,
        context=torch.randn(1, 1, gan.generator.cross_attention_dim),
        class_labels=cl,
    )

    gan.discriminator(gen_output)

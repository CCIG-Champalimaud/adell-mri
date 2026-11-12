import pytest
import torch

from adell_mri.modules.gan.generator import (
    AttnMidBlock,
    CrossAttnMidBlock,
    Generator,
)


def _generator_kwargs(spatial_dims):
    return {
        "spatial_dims": spatial_dims,
        "in_channels": 1,
        "out_channels": 1,
        "num_channels": (8, 8),
        "num_res_blocks": 1,
        "attention_levels": (False, False),
        "norm_num_groups": 8,
        "strides": (2, 1),
    }


@pytest.mark.parametrize("spatial_dims", [2, 3])
@pytest.mark.parametrize("no_skip_connection", [False, True])
def test_generator_latent_contract(spatial_dims, no_skip_connection):
    model = Generator(
        **_generator_kwargs(spatial_dims),
        no_skip_connection=no_skip_connection,
        out_activation=torch.nn.Identity,
    )
    size = 16
    x = torch.rand(2, 1, *([size] * spatial_dims))

    output = model(x)
    latent = model.encode(x)

    assert output.shape == x.shape
    assert latent.shape[0] == 2
    assert latent.shape[1] == model.block_out_channels[-1]
    assert tuple(model.strides[-1]) == (1,) * spatial_dims
    assert model.latent_downsample_factor == 2
    assert isinstance(model.out_activation, torch.nn.Identity)

    if no_skip_connection:
        decoded = model.decode(latent)
        assert decoded.shape == x.shape
    else:
        with pytest.raises(ValueError):
            model.decode(latent)


@pytest.mark.parametrize("strides", [(2, 2), (2, 1)])
def test_generator_resblock_updown_strides(strides):
    kwargs = _generator_kwargs(2)
    kwargs["strides"] = strides
    model = Generator(
        **kwargs,
        resblock_updown=True,
        no_skip_connection=True,
    )
    x = torch.rand(1, 1, 16, 16)

    assert model(x).shape == x.shape


@pytest.mark.parametrize("spatial_dims", [2, 3])
def test_generator_class_conditioning(spatial_dims):
    model = Generator(
        **_generator_kwargs(spatial_dims),
        num_class_embeds=4,
    )
    x = torch.rand(2, 1, *([16] * spatial_dims))
    labels = torch.tensor([0, 3])

    output = model(x, class_labels=labels)

    assert output.shape == x.shape
    assert model.class_embedding.num_embeddings == 4


@pytest.mark.parametrize("spatial_dims", [2, 3])
def test_generator_cross_attention_mid_block(spatial_dims):
    model = Generator(
        **_generator_kwargs(spatial_dims),
        with_conditioning=True,
        cross_attention_dim=8,
    )
    x = torch.rand(2, 1, *([16] * spatial_dims))
    context = torch.rand(2, 1, 8)

    output = model(x, context=context)

    assert output.shape == x.shape
    assert isinstance(model.middle_block, CrossAttnMidBlock)


@pytest.mark.parametrize("spatial_dims", [2, 3])
def test_generator_self_attention_mid_block(spatial_dims):
    model = Generator(**_generator_kwargs(spatial_dims))

    assert isinstance(model.middle_block, AttnMidBlock)

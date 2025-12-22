import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from adell_mri.modules.layers.linear_blocks import (
    MultiHeadSelfAttention,
    SelfAttention,
)

input_dim_primary = 32
input_dim_context = 16
attention_dim = 64
hidden_dim = 96
output_dim = 128
batch_size = 4
token_size = 9
n_heads = 2


def test_self_attention():
    out = SelfAttention(input_dim_primary, attention_dim, hidden_dim)(
        torch.rand(size=[batch_size, token_size, input_dim_primary])
    )
    assert list(out.shape) == [batch_size, token_size, hidden_dim]


def test_multi_head_attention():
    out = MultiHeadSelfAttention(
        input_dim_primary,
        attention_dim,
        hidden_dim,
        output_dim,
        n_heads=n_heads,
    )(torch.rand(size=[batch_size, token_size, input_dim_primary]))
    assert list(out.shape) == [batch_size, token_size, output_dim]


def test_windowed_multi_head_attention():
    out = MultiHeadSelfAttention(
        input_dim_primary,
        attention_dim,
        hidden_dim,
        output_dim,
        n_heads=n_heads,
        window_size=[8, 8, 8],
    )(torch.rand(size=[batch_size, token_size, input_dim_primary]))
    assert list(out.shape) == [batch_size, token_size, output_dim]


def test_windowed_multi_head_attention_irregular_shape():
    out = MultiHeadSelfAttention(
        input_dim_primary,
        attention_dim,
        hidden_dim,
        output_dim,
        n_heads=n_heads,
        window_size=[8, 8, 8],
    )(torch.rand(size=[batch_size, 4, token_size, input_dim_primary]))
    assert list(out.shape) == [batch_size, 4, token_size, output_dim]

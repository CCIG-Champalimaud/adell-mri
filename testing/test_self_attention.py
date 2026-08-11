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


def _manual_attention(module, X, mask=None):
    """Manual attention implementation matching the SDPA reference semantics."""
    sh = X.shape
    b, t, _ = sh[:-2], sh[-2], sh[-1]
    QKV = module.qkv(X)
    pd = [*[i for i in range(len(b))], len(b) + 1, len(b), len(b) + 2]
    QKV = QKV.reshape(
        *b, t, module.n_heads, module.qkv_dim // module.n_heads
    ).permute(*pd)
    Q, K, V = (
        QKV[..., module.q_idx],
        QKV[..., module.k_idx],
        QKV[..., module.v_idx],
    )
    Q = module.q_norm(Q)
    K = module.k_norm(K)
    S = Q @ torch.transpose(K, -1, -2)
    S = S / module.reg_const
    if module.window_size:
        rpb = module.relative_position_bias_table[
            module.relative_position_index.clone()[:t, :t].reshape(-1)
        ].reshape(module.n_heads, t, t)
        S = S + rpb
    if mask is not None:
        m = mask.to(S)
        if m.ndim == 3:
            m = m.unsqueeze(1)
        S = S + m
    S = module.drop_op(module.sm(S))
    V_tilde = S @ V
    V_tilde = V_tilde.transpose(-3, -2).reshape(*b, t, module.hidden_dim)
    return module.output_layer(V_tilde)


def _make_mha(**kwargs):
    defaults = dict(
        input_dim=input_dim_primary,
        attention_dim=attention_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_heads=n_heads,
        dropout_rate=0.0,
    )
    defaults.update(kwargs)
    return MultiHeadSelfAttention(**defaults).eval()


def _assert_close(a, b, atol=1e-5, rtol=1e-5):
    assert torch.allclose(
        a, b, atol=atol, rtol=rtol
    ), f"max diff {(a - b).abs().max().item()}"


# --- numerics: SDPA output matches the manual attention math ---


def test_numerics_plain():
    mha = _make_mha()
    X = torch.randn(batch_size, token_size, input_dim_primary)
    with torch.no_grad():
        _assert_close(mha(X), _manual_attention(mha, X))


def test_numerics_multidimensional_batch():
    mha = _make_mha()
    X = torch.randn(2, batch_size, token_size, input_dim_primary)
    with torch.no_grad():
        _assert_close(mha(X), _manual_attention(mha, X))


def test_numerics_windowed():
    # t must match the window area for the relative position bias reshape
    t = 16
    mha = _make_mha(window_size=[4, 4])
    X = torch.randn(batch_size, t, input_dim_primary)
    with torch.no_grad():
        _assert_close(mha(X), _manual_attention(mha, X))


def test_numerics_2d_mask():
    mha = _make_mha()
    t = token_size
    X = torch.randn(batch_size, t, input_dim_primary)
    mask = torch.zeros(t, t)
    mask[t // 2 :, t // 2 :] = float("-100.0")
    with torch.no_grad():
        _assert_close(mha(X, mask=mask), _manual_attention(mha, X, mask=mask))


def test_numerics_4d_mask():
    mha = _make_mha()
    t = token_size
    X = torch.randn(batch_size, t, input_dim_primary)
    mask = torch.zeros(batch_size, 1, t, t)
    mask[:, :, t // 2 :, t // 2 :] = float("-100.0")
    with torch.no_grad():
        _assert_close(mha(X, mask=mask), _manual_attention(mha, X, mask=mask))


def test_numerics_windowed_3d_mask():
    mha = _make_mha(window_size=[4, 4])
    t = 16
    X = torch.randn(batch_size, t, input_dim_primary)
    mask = torch.zeros(batch_size, t, t)
    mask[:, t // 2 :, t // 2 :] = float("-100.0")
    with torch.no_grad():
        _assert_close(mha(X, mask=mask), _manual_attention(mha, X, mask=mask))


# --- regression: gradients flow and match the manual implementation ---


def test_backward_produces_gradients():
    mha = _make_mha()
    X = torch.randn(batch_size, token_size, input_dim_primary)
    out = mha(X)
    out.sum().backward()
    for name, param in mha.named_parameters():
        assert param.grad is not None, f"missing gradient for {name}"


def test_numerics_gradients():
    mha = _make_mha()
    X = torch.randn(batch_size, token_size, input_dim_primary)
    out = mha(X)
    ref = _manual_attention(mha, X)
    loss = (out - ref).pow(2).sum()
    loss.backward()
    for name, param in mha.named_parameters():
        assert param.grad is not None, f"missing gradient for {name}"
        assert (
            param.grad.abs().sum() < 1e-3
        ), f"gradient unexpectedly large for {name}"


def test_2d_mask_actually_masks():
    mha = _make_mha()
    t = token_size
    X = torch.randn(batch_size, t, input_dim_primary)
    mask = torch.zeros(t, t)
    mask[:, t // 2 :] = float("-100.0")
    X_a = X.clone()
    X_b = X.clone()
    X_b[:, t // 2 :, :] = torch.randn_like(X_b[:, t // 2 :, :])
    with torch.no_grad():
        out_a = mha(X_a, mask=mask)
        out_b = mha(X_b, mask=mask)
    # masked query tokens attend only to the unmasked region, so their
    # outputs are unaffected by the content of the masked tokens
    _assert_close(out_a[:, : t // 2, :], out_b[:, : t // 2, :], atol=1e-6)
    assert (out_a[:, t // 2 :, :] - out_b[:, t // 2 :, :]).abs().max() > 1e-3

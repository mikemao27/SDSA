"""
Correctness tests for SpikingSelfAttention's two attention_mode groupings.

Since spike-driven attention has no softmax between the two matmuls, (Q @ K^T) @ V and Q @ (K^T @ V) are mathematically identical (matrix-multiply
associativity), and the scalar attention_scale commutes through either grouping. These tests verify that identity holds both at the raw-tensor level
and through the actual SpikingSelfAttention module (with weights synced between a "quadratic" and a "linear" instance).
"""

import torch

from transformer.attention import SpikingSelfAttention

def test_raw_matmul_associativity() -> None:
    """
    (Q @ K^T) @ V and Q @ (K^T @ V) must match within floating-point tolerance for random binary Q, K, V.
    """
    torch.manual_seed(0)
    T, B, H, N, d = 3, 2, 4, 10, 8
    scale = 0.125

    Q = torch.bernoulli(torch.full((T, B, H, N, d), 0.5))
    K = torch.bernoulli(torch.full((T, B, H, N, d), 0.5))
    V = torch.bernoulli(torch.full((T, B, H, N, d), 0.5))

    quadratic = ((Q @ K.transpose(-2, -1)) * scale) @ V
    linear = (Q @ (K.transpose(-2, -1) @ V)) * scale

    assert torch.allclose(quadratic, linear, atol=1e-5)

def test_module_level_equivalence() -> None:
    """
    Two SpikingSelfAttention instances with identical weights but different attention_mode must produce identical output for the same input.
    """
    torch.manual_seed(0)
    embed_dim, num_heads = 16, 4
    T, B, N = 3, 2, 5

    quadratic_attn = SpikingSelfAttention(embed_dim, num_heads, attention_mode="quadratic")
    linear_attn = SpikingSelfAttention(embed_dim, num_heads, attention_mode="linear")
    linear_attn.load_state_dict(quadratic_attn.state_dict())

    quadratic_attn.eval()
    linear_attn.eval()

    x = torch.randn(T, B, N, embed_dim)

    with torch.no_grad():
        quadratic_out = quadratic_attn(x)
        linear_out = linear_attn(x)

    assert torch.allclose(quadratic_out, linear_out, atol=1e-5)

def test_invalid_attention_mode_raises() -> None:
    """
    Constructing SpikingSelfAttention with an unrecognized attention_mode must raise ValueError.
    """
    try:
        SpikingSelfAttention(16, 4, attention_mode = "bogus")
    except ValueError:
        return
    raise AssertionError("Expected ValueError for an invalid attention_mode.")
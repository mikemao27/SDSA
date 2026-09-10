"""
Dense (non-spiking) transformer baseline: a standard, matched-parameter ANN counterpart to SpikingTransformer, sharing the same Tokenizer
interface (via its embed_static method) so both models see identically-tokenized input. Uses ordinary ViT ingredients, real LayerNorm, softmax
attention, GELU, rather than being artificially handicapped to resemble the spiking model, so the accuracy/energy/latency comparison in
docs/report.md reflects a fair fight between architectures, not a strawman.
"""

import torch
import torch.nn as nn

from .tokenizers import Tokenizer

class DenseSelfAttention(nn.Module):
    """
    Standard multi-head softmax self-attention over a (B, N, embed_dim) token sequence. embed_dim is the token embedding dimension. num_heads is
    the number of attention heads, must evenly divide embed_dim. qkv_bias is whether the Q/K/V linear projection uses a bias term.
    """
    def __init__(self, embed_dim: int, num_heads: int, qkv_bias: bool = True) -> None:
        assert embed_dim % num_heads == 0, f"num_heads must be a factor of embed_dim; we got embed_dim: {embed_dim} and num_heads: {num_heads} instead."

        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias = qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is a tensor of shape (B, N, embed_dim). Returns a tensor of shape (B, N, embed_dim).
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # each (B, num_heads, N, head_dim)

        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim = -1)

        output = attention @ v # (B, num_heads, N, head_dim)
        output = output.transpose(1, 2).reshape(B, N, C)

        return self.proj(output)

class DenseMLP(nn.Module):
    """
    Standard two-layer feed-forward block with a GELU nonlinearity. embed_dim is the input/output feature dimension. hidden_dim is the hidden
    layer dimension (typically a multiple of embed_dim).
    """
    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is a tensor of shape (B, N, embed_dim). Returns a tensor of shape (B, N, embed_dim).
        """
        return self.fc2(self.act(self.fc1(x)))

class DenseTransformerBlock(nn.Module):
    """
    One standard pre-norm transformer block: LayerNorm -> softmax self-attention -> residual, then LayerNorm -> MLP -> residual. embed_dim is the
    token embedding dimension. num_heads is the number of attention heads. mlp_ratio is the hidden-dim multiplier for the MLP.
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = DenseSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = DenseMLP(embed_dim, int(embed_dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is a tensor of shape (B, N, embed_dim). Returns a tensor of shape (B, N, embed_dim).
        """
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class DenseViT(nn.Module):
    """
    A standard dense Vision Transformer, sharing the Tokenizer abstraction with SpikingTransformer (see transformer/tokenizers.py) so both
    models see identically-tokenized input: this uses tokenizer.embed_static(...), since a dense model has no time dimension. num_classes is
    the number of output classes. embed_dim, depth, num_heads, mlp_ratio mirror SpikingTransformer's corresponding arguments, so a DenseViT and a
    SpikingTransformer built from the same config are matched in architecture.
    """
    def __init__(
        self,
        tokenizer: Tokenizer,
        num_classes: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        assert tokenizer.embed_dim == embed_dim, (
            f"tokenizer.embed_dim ({tokenizer.embed_dim}) must match embed_dim ({embed_dim})."
        )

        super().__init__()
        self.tokenizer = tokenizer
        self.blocks = nn.ModuleList([DenseTransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, raw_input: torch.Tensor) -> torch.Tensor:
        """
        Classify a batch of raw, modality-specific input (same raw_input contract as SpikingTransformer.forward, minus the time dimension).
        Returns a logits tensor of shape (B, num_classes).
        """
        tokens = self.tokenizer.embed_static(raw_input)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        pooled = tokens.mean(dim = 1)

        return self.head(pooled)
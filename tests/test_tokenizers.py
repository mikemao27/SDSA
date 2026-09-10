"""
Shape correctness tests for ImagePatchTokenizer and SHDTokenizer.
"""

import torch

from transformer.tokenizers import ImagePatchTokenizer, SHDTokenizer

def test_image_tokenizer_repeat_coding_mnist_shape() -> None:
    """
    MNIST-shaped input (B, 1, 28, 28) with patch_size=7 should produce (T, B, 16, embed_dim) tokens (N = (28/7)^2 = 16), and (B, 16, embed_dim)
    for embed_static.
    """
    tokenizer = ImagePatchTokenizer(image_size = 28, patch_size = 7, in_channels = 1, embed_dim = 32, num_timesteps = 4, coding = "repeat")
    x = torch.rand(2, 1, 28, 28)

    tokens = tokenizer(x)
    assert tokens.shape == (4, 2, 16, 32)

    static_tokens = tokenizer.embed_static(x)
    assert static_tokens.shape == (2, 16, 32)

def test_image_tokenizer_rate_coding_cifar_shape() -> None:
    """
    CIFAR-shaped input (B, 3, 32, 32) with patch_size = 8 and coding = "rate" should produce (T, B, 16, embed_dim) tokens (N = (32/8)^2 = 16), and the
    tokens must be a real function of T (rate coding injects per-timestep randomness before the conv, so timesteps should generally differ).
    """
    torch.manual_seed(0)
    tokenizer = ImagePatchTokenizer(image_size = 32, patch_size = 8, in_channels = 3, embed_dim = 16, num_timesteps = 5, coding = "rate")
    x = torch.rand(2, 3, 32, 32)

    tokens = tokenizer(x)
    assert tokens.shape == (5, 2, 16, 16)
    assert not torch.allclose(tokens[0], tokens[1])

def test_shd_tokenizer_shape() -> None:
    """
    SHD-shaped input (B, num_tokens, num_channels) should produce (T, B, num_tokens, embed_dim) tokens, and (B, num_tokens, embed_dim) for
    embed_static.
    """
    tokenizer = SHDTokenizer(num_channels = 700, num_tokens = 16, embed_dim = 32, num_timesteps = 4)
    x = torch.rand(2, 16, 700)

    tokens = tokenizer(x)
    assert tokens.shape == (4, 2, 16, 32)

    static_tokens = tokenizer.embed_static(x)
    assert static_tokens.shape == (2, 16, 32)

def test_shd_tokenizer_repeats_identically_across_timesteps() -> None:
    """
    SHDTokenizer only supports "repeat" coding, so every timestep's currents should be identical (no per-timestep randomness is injected).
    """
    tokenizer = SHDTokenizer(num_channels = 700, num_tokens = 16, embed_dim = 32, num_timesteps = 4)
    x = torch.rand(2, 16, 700)

    tokens = tokenizer(x)
    for t in range(1, tokens.shape[0]):
        assert torch.equal(tokens[0], tokens[t])

def test_image_tokenizer_rejects_invalid_coding() -> None:
    """
    Constructing ImagePatchTokenizer with an unrecognized coding must raise ValueError.
    """
    try:
        ImagePatchTokenizer(image_size = 28, patch_size = 7, in_channels = 1, embed_dim = 16, num_timesteps = 4, coding = "bogus")
    except ValueError:
        return
    raise AssertionError("Expected ValueError for an invalid coding.")
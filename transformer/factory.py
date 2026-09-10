"""
Factories for building a Tokenizer or a full model from a SpikingTransformerConfig, keeping all dataset/model_family branching in one place
instead of scattered across model.py or call sites like train.py.
"""

import torch.nn as nn

from .baseline import DenseViT
from .model import SpikingTransformer
from .tokenizers import ImagePatchTokenizer, SHDTokenizer, Tokenizer

def build_tokenizer(config) -> Tokenizer:
    """
    Construct the Tokenizer appropriate for config.dataset. config is a SpikingTransformerConfig (see config.py).
    """
    if config.dataset in {"mnist", "cifar10"}:
        return ImagePatchTokenizer(
            image_size = config.image_size,
            patch_size = config.patch_size,
            in_channels = config.in_channels,
            embed_dim = config.embed_dim,
            num_timesteps = config.num_timesteps,
            coding = config.coding,
        )

    if config.dataset == "shd":
        return SHDTokenizer(
            num_channels = config.shd_num_channels,
            num_tokens = config.shd_num_tokens,
            embed_dim = config.embed_dim,
            num_timesteps = config.num_timesteps,
        )

    raise ValueError(f"Unrecognized dataset: {config.dataset!r}")

def build_model(config) -> nn.Module:
    """
    Construct the full model (a SpikingTransformer, or a dense ViT baseline) appropriate for config.model_family,
    wired up with the tokenizer config.dataset calls for. config is a SpikingTransformerConfig (see config.py).
    """
    tokenizer = build_tokenizer(config)

    if config.model_family == "snn":
        return SpikingTransformer(
            tokenizer = tokenizer,
            num_classes = config.num_classes,
            embed_dim = config.embed_dim,
            depth = config.depth,
            num_heads = config.num_heads,
            mlp_ratio = config.mlp_ratio,
            threshold = config.threshold,
            beta = config.beta,
            attention_mode = config.attention_mode,
            neuron_type = config.neuron_type,
            track_firing_rate = config.track_firing_rate,
        )

    if config.model_family == "ann":
        return DenseViT(
            tokenizer = tokenizer,
            num_classes = config.num_classes,
            embed_dim = config.embed_dim,
            depth = config.depth,
            num_heads = config.num_heads,
            mlp_ratio = config.mlp_ratio,
        )

    raise ValueError(f"Unrecognized model_family: {config.model_family!r}")
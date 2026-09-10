"""
Hyperparameter configuration for the spiking transformer, across its supported datasets (MNIST, CIFAR-10, SHD) and model families (the spiking
SNN, or the dense ANN baseline added in Stage 5).

Rather than constructing SpikingTransformerConfig directly with every field spelled out, prefer the preset builder functions at the bottom of
this module (mnist_config(), cifar10_config(), shd_config()): each fills in sensible, dataset-appropriate defaults for the fields that don't
apply to other datasets, and accepts **overrides for anything you want to change (e.g. mnist_config(num_epochs = 1) for a quick smoke test).
"""

from dataclasses import dataclass, replace

@dataclass
class SpikingTransformerConfig:
    """
    All hyperparameters needed to build, train, and evaluate the model. Most fields are given defaults so ad-hoc construction stays convenient,
    but prefer the mnist_config()/cifar10_config()/shd_config() presets below over relying on these bare defaults.
    """

    # Dataset selection:
    dataset: str = "mnist" # "mnist" | "cifar10" | "shd".

    # Model architecture (image datasets only; ignored for "shd"):
    image_size: int = 28 # The input image height/width, e.g. 28 for MNIST, 32 for CIFAR-10.
    patch_size: int = 7 # The patch height/width; must evenly divide image_size.
    in_channels: int = 1 # The number of input image channels (1 for MNIST, 3 for CIFAR-10).

    # Model architecture (SHD only; ignored for image datasets):
    shd_num_channels: int = 700 # SHD's fixed sensor channel count.
    shd_num_tokens: int = 16 # Number of temporal chunks SHD's raw spike-time events are binned into (see datasets.py::get_shd_dataloaders); this is N.

    # Shared architecture:
    num_classes: int = 10 # 10 for MNIST/CIFAR-10; 20 for SHD (10 spoken digits x English/German).
    embed_dim: int = 64 # The token embedding dimension.
    depth: int = 2 # The number of stacked SpikingTransformerBlock layers.
    num_heads: int = 2 # The attention heads per block; must evenly divide embed_dim.
    mlp_ratio: float = 4.0 # SpikingMLP hidden_dim = embed_dim * mlp_ratio.

    # Spiking neuron dynamics:
    num_timesteps: int = 8 # T, number of simulation time-steps per forward pass.
    threshold: float = 1.0 # The LIF firing threshold.
    beta: float = 0.9 # The LIF leak/decay factor, in (0, 1).
    surrogate_alpha: float = 2.0 # The surrogate gradient steepness (see transformer/surrogate.py).
    coding: str = "repeat" # "repeat" or "rate" (image datasets only: SHD is already event-based, so only "repeat" is valid for it).

    # Training:
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 5
    device: str = "cpu" # "cpu" or "mps" (Apple Silicon GPU).

    # Ablatable architecture/neuron choices:
    model_family: str = "snn" # "snn" (SpikingTransformer) | "ann" (dense ViT baseline, added in Stage 5).
    attention_mode: str = "linear" # Either "linear" (O(N) spike-driven attention) or "quadratic" (O(N^2), kept for comparison/ablation).
    neuron_type: str = "lif" # Spiking neuron implementation, e.g. "lif" or "plif" (see transformer/neurons.py::build_neuron).
    track_firing_rate: bool = False # If True, every LIF neuron records its mean firing rate for energy accounting (see transformer/energy.py).

    def __post_init__(self) -> None:
        if self.dataset not in {"mnist", "cifar10", "shd"}:
            raise ValueError(f"dataset must be one of 'mnist', 'cifar10', 'shd'; we got {self.dataset!r} instead.")
        if self.coding not in {"repeat", "rate"}:
            raise ValueError(f"coding must be either 'repeat' or 'rate'; we got {self.coding!r} instead.")
        if self.dataset == "shd" and self.coding == "rate":
            raise ValueError(
                "SHD is already a genuinely event-based dataset (its raw spike-time events are binned into discrete per-chunk spike counts); "
                "rate-coding those counts is not well-defined the way [0, 1] pixel intensities are. Use coding='repeat' for SHD."
            )
        if self.model_family not in {"snn", "ann"}:
            raise ValueError(f"model_family must be either 'snn' or 'ann'; we got {self.model_family!r} instead.")

def mnist_config(**overrides) -> SpikingTransformerConfig:
    """
    A SpikingTransformerConfig preset for MNIST, with architecture/training defaults sized to comfortably finish a full headline run (full
    dataset, ~10 epochs) well under an hour on a modern multi-core CPU. Pass keyword overrides for anything you want to change.
    """
    defaults = dict(
        dataset = "mnist",
        image_size = 28,
        patch_size = 7,
        in_channels = 1,
        num_classes = 10,
        embed_dim = 128,
        depth = 3,
        num_heads = 4,
        mlp_ratio = 4.0,
        num_timesteps = 4,
        threshold = 1.0,
        beta = 0.9,
        surrogate_alpha = 2.0,
        batch_size = 128,
        learning_rate = 1e-3,
        num_epochs = 10,
        device = "cpu",
    )
    defaults.update(overrides)
    return SpikingTransformerConfig(**defaults)

def cifar10_config(**overrides) -> SpikingTransformerConfig:
    """
    A SpikingTransformerConfig preset for CIFAR-10. patch_size=8 keeps the token count (N = 16) comparable to the MNIST preset's, so energy/latency
    comparisons across datasets aren't confounded by very different sequence lengths. Pass keyword overrides for anything you want to change.
    """
    defaults = dict(
        dataset = "cifar10",
        image_size = 32,
        patch_size = 8,
        in_channels = 3,
        num_classes = 10,
        embed_dim = 128,
        depth = 3,
        num_heads = 4,
        mlp_ratio = 4.0,
        num_timesteps = 4,
        threshold = 1.0,
        beta = 0.9,
        surrogate_alpha = 2.0,
        batch_size = 128,
        learning_rate = 1e-3,
        num_epochs = 10,
        device = "cpu",
    )
    defaults.update(overrides)
    return SpikingTransformerConfig(**defaults)

def shd_config(**overrides) -> SpikingTransformerConfig:
    """
    A SpikingTransformerConfig preset for Spiking Heidelberg Digits (SHD): 20 classes (10 spoken digits, English and German), 16 temporal-chunk
    tokens (matching the other presets' N=16 for comparability), more epochs than the image presets since SHD's training set is much smaller
    (~8k samples). Pass keyword overrides for anything you want to change.
    """
    defaults = dict(
        dataset = "shd",
        shd_num_channels = 700,
        shd_num_tokens = 16,
        num_classes = 20,
        embed_dim = 128,
        depth = 3,
        num_heads = 4,
        mlp_ratio = 4.0,
        num_timesteps = 4,
        threshold = 1.0,
        beta = 0.9,
        surrogate_alpha = 2.0,
        coding = "repeat",
        batch_size = 128,
        learning_rate = 1e-3,
        num_epochs = 20,
        device = "cpu",
    )
    defaults.update(overrides)
    return SpikingTransformerConfig(**defaults)

_DATASET_PRESETS = {"mnist": mnist_config, "cifar10": cifar10_config, "shd": shd_config}

def ann_baseline_config(dataset: str, **overrides) -> SpikingTransformerConfig:
    """
    Build a dense-ANN-baseline config matched in architecture (embed_dim, depth, num_heads, mlp_ratio, batch_size, learning_rate, num_epochs) to
    the given dataset's SNN preset (mnist_config/cifar10_config/shd_config), but with model_family = "ann". SNN-only fields (threshold, beta,
    num_timesteps, neuron_type, attention_mode, coding, track_firing_rate) are simply carried over from the SNN preset's defaults; build_model
    ignores them for an "ann" model_family. Pass keyword overrides for anything you want to change.
    """
    if dataset not in _DATASET_PRESETS:
        raise ValueError(f"dataset must be one of {sorted(_DATASET_PRESETS)}; we got {dataset!r} instead.")

    base_config = _DATASET_PRESETS[dataset]()
    return replace(base_config, model_family = "ann", **overrides)

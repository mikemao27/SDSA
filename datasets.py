"""
Dataset loaders for the three supported modalities: MNIST and CIFAR-10 (both images, via torchvision), and SHD (Spiking Heidelberg Digits, a
genuinely event-based neuromorphic audio dataset, via the `tonic` package). Each returns a (train_loader, test_loader) pair of DataLoaders whose
batches are directly consumable by the tokenizer config.dataset selects (see transformer/factory.py::build_tokenizer).
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import SpikingTransformerConfig

def get_mnist_dataloaders(config: SpikingTransformerConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Build MNIST train/test DataLoaders. Batches are (images, labels) with images of shape (B, 1, 28, 28), pixel values in [0, 1].
    """
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(root = "./data", train = True, download = True, transform = transform)
    test_dataset = datasets.MNIST(root = "./data", train = False, download = True, transform = transform)

    train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle = False)

    return train_loader, test_loader

def get_cifar10_dataloaders(config: SpikingTransformerConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Build CIFAR-10 train/test DataLoaders. Batches are (images, labels) with images of shape (B, 3, 32, 32), pixel values in [0, 1].

    Deliberately no mean/std normalization: rate coding (config.coding == "rate") requires pixel values in [0, 1] to serve as Bernoulli
    probabilities, and normalizing would break that invariant. The dense ANN baseline (Stage 5) uses this same unnormalized transform, so the
    accuracy/energy comparison isolates the architectural difference rather than confounding it with a preprocessing advantage: this does mean
    the ANN baseline will underperform a fully-tuned, normalized CIFAR-10 ViT (see docs/report.md's limitations).
    """
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.CIFAR10(root = "./data", train = True, download = True, transform = transform)
    test_dataset = datasets.CIFAR10(root = "./data", train = False, download = True, transform = transform)

    train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle = False)

    return train_loader, test_loader

def _shd_collate(batch):
    """
    Collate a batch of (frame, label) pairs from tonic's SHD dataset (each frame a numpy array of shape (n_time_bins, 1, num_channels), each
    label a numpy/python int scalar) into a single (x, y) tensor pair: x of shape (B, n_time_bins, num_channels), y of shape (B,).
    """
    frames, labels = zip(*batch)
    x = torch.stack([torch.from_numpy(frame.reshape(frame.shape[0], -1)).float() for frame in frames], dim = 0)
    y = torch.tensor(labels, dtype = torch.long)
    return x, y

def get_shd_dataloaders(config: SpikingTransformerConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Build Spiking Heidelberg Digits (SHD) train/test DataLoaders, via the `tonic` package. Each raw sample is a variable-length stream of
    (time, channel) spike events over config.shd_num_channels channels; tonic.transforms.ToFrame bins each sample's events into exactly
    config.shd_num_tokens fixed-size temporal chunks (summed spike counts per chunk per channel), which is what makes samples fixed-size and
    lets the standard DataLoader/collate machinery batch them.

    Batches are (x, y) with x of shape (B, shd_num_tokens, shd_num_channels) (float spike counts, not necessarily binary) and y of shape (B,)
    (int64 class labels in [0, 20), 10 spoken digits x English/German). Requires `pip install tonic`; raises ImportError with guidance if it's
    not installed. The dataset is downloaded (from a Zenodo mirror) and cached under ./data on first use.
    """
    try:
        import tonic
    except ImportError as exc:
        raise ImportError("The 'tonic' package is required for the SHD dataset; install it with `pip install tonic`.") from exc

    sensor_size = (config.shd_num_channels, 1, 1)
    frame_transform = tonic.transforms.ToFrame(sensor_size = sensor_size, n_time_bins = config.shd_num_tokens)

    train_dataset = tonic.datasets.SHD(save_to = "./data", train = True, transform = frame_transform)
    test_dataset = tonic.datasets.SHD(save_to = "./data", train = False, transform = frame_transform)

    train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True, collate_fn = _shd_collate)
    test_loader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle = False, collate_fn = _shd_collate)

    return train_loader, test_loader

def get_dataloaders(config: SpikingTransformerConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Dispatch to the DataLoader-building function appropriate for config.dataset ("mnist", "cifar10", or "shd").
    """
    if config.dataset == "mnist":
        return get_mnist_dataloaders(config)
    if config.dataset == "cifar10":
        return get_cifar10_dataloaders(config)
    if config.dataset == "shd":
        return get_shd_dataloaders(config)

    raise ValueError(f"Unrecognized dataset: {config.dataset!r}")
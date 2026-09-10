"""
Training script for the spiking transformer.

Builds a dataset-appropriate config (see config.py's mnist_config/cifar10_config/shd_config presets), loads the corresponding DataLoaders (see
datasets.py), builds the model (see transformer/factory.py::build_model), and runs a standard supervised training loop (cross-entropy on the
model's output logits, which are read out from accumulated spiking activity over the simulated timesteps).

To run this, enter the following command in terminal: "python train.py" (defaults to MNIST), or "python train.py --dataset cifar10" /
"python train.py --dataset shd" for the other supported datasets.
"""

import argparse

import torch
from torch.utils.data import DataLoader

from config import cifar10_config, mnist_config, shd_config, SpikingTransformerConfig
from datasets import get_dataloaders
from transformer.factory import build_model

_CONFIG_PRESETS = {
    "mnist": mnist_config,
    "cifar10": cifar10_config,
    "shd": shd_config,
}

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments: --dataset selects which config preset (and DataLoaders) to use.
    """
    parser = argparse.ArgumentParser(description = "Train a SpikingTransformer.")
    parser.add_argument("--dataset", type = str, default = "mnist", choices = sorted(_CONFIG_PRESETS), help = "Which dataset preset to train on.")
    parser.add_argument("--num-epochs", type = int, default = None, help = "Override the preset's default number of epochs.")
    return parser.parse_args()

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Run one training epoch over the loader. model is the model being trained. loader is the training data loader. optimizer is the optimizer for
    the model's parameters. device is the device to run computation on. Returns the average training loss over the epoch.
    """
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    Evaluate classification accuracy of model on loader. model is the model being evaluated. loader is the evaluation data loader (typically the
    test set). device is the device to run computation on. Returns the classification accuracy, in [0.0, 1.0].
    """
    model.eval()
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        preds = logits.argmax(dim = 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total

def build_config(dataset: str, num_epochs: int | None) -> SpikingTransformerConfig:
    """
    Build the SpikingTransformerConfig preset for dataset, applying a num_epochs override if given.
    """
    overrides = {} if num_epochs is None else {"num_epochs": num_epochs}
    return _CONFIG_PRESETS[dataset](**overrides)

def main() -> None:
    """
    Wire together config, data, model, optimizer, and the training loop.
    """
    args = parse_args()
    config = build_config(args.dataset, args.num_epochs)

    device = torch.device(config.device if (config.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    print(f"Dataset: {config.dataset}, using device: {device}")

    train_loader, test_loader = get_dataloaders(config)
    model = build_model(config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)

    for epoch in range(config.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch + 1}/{config.num_epochs} training_loss = {train_loss:.4f} test_accuracy = {test_acc:.4f}")

    torch.save(model.state_dict(), "checkpoint.pt")
    print("Saved checkpoint to checkpoint.pt")

if __name__ == "__main__":
    main()
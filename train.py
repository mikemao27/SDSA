"""
Training script for the spiking transformer MNIST proof-of-concept.

Loads MNIST, builds a SpikingTransformer from a SpikingTransformerConfig, and runs a standard supervised training loop (cross-entropy on the model's
output logits, which are read out from accumulated spiking activity over the simulated timesteps).

To run this, enter the following command in terminal: "python train.py".
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import SpikingTransformerConfig
from transformer.model import SpikingTransformer

def get_dataloaders(config: SpikingTransformerConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Build MNIST train/test DataLoaders. config is the run configuration (used for batch_size, and any normalization choices you decide on). Returns 
    a (train_loader, test_loader) tuple.
    """
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(root = "./data", train = True, download = True, transform = transform)
    test_dataset = datasets.MNIST(root = "./data", train = False, download = True, transform = transform)

    train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle = False)

    return train_loader, test_loader

def train_one_epoch(
    model: SpikingTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Run one training epoch over the loader. model is the spiking transformer being trained. loader is the training data loader. optimizer is the 
    optimizer for the model's parameters. device is the device to run computation on. Returns the average training loss over the epoch.
    """
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model: SpikingTransformer, loader: DataLoader, device: torch.device) -> float:
    """
    Evaluate classification accuracy of model on loader. model is the spiking transformer being evaluated. loader is the evaluation data loader 
    (typically the test set). device is the device to run computation on. Returns the classification accuracy, in [0.0, 1.0].
    """
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total

def main() -> None:
    """
    Wire together config, data, model, optimizer, and the training loop.
    """
    config = SpikingTransformerConfig(
        image_size = 28,
        patch_size = 7,
        in_channels = 1,
        num_classes = 10,
        embed_dim = 64,
        depth = 2,
        num_heads = 2,
        mlp_ratio = 4.0,
        num_timesteps = 8,
        threshold = 1.0,
        beta = 0.9,
        surrogate_alpha = 2.0,
        batch_size = 64,
        learning_rate = 1e-3,
        num_epochs = 5,
        device = "mps",
    )

    device = torch.device(config.device if (config.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(config)

    model = SpikingTransformer(
        image_size = config.image_size,
        patch_size = config.patch_size,
        in_channels = config.in_channels,
        num_classes = config.num_classes,
        embed_dim = config.embed_dim,
        depth = config.depth,
        num_heads = config.num_heads,
        mlp_ratio = config.mlp_ratio,
        num_timesteps = config.num_timesteps,
        threshold = config.threshold,
        beta = config.beta,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)

    for epoch in range(config.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch + 1}/{config.num_epochs} training_loss = {train_loss:.4f} test_accuracy = {test_acc:.4f}")

    torch.save(model.state_dict(), "checkpoint.pt")
    print("Saved checkpoint to checkpoint.pt")

if __name__ == "__main__":
    main()
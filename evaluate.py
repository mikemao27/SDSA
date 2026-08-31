"""
Standalone evaluation script: load a trained checkpoint and report test accuracy.

To run this, enter the following command in terminal: "python evaluate.py --checkpoint path/to/checkpoint.pt".
"""

import argparse

import torch

from config import SpikingTransformerConfig
from transformer.model import SpikingTransformer

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments (at minimum, a --checkpoint path).
    """
    parser = argparse.ArgumentParser(description = "Evaluate a trained SpikingTransformer checkpoint on MNIST.")
    parser.add_argument("--checkpoint", type = str, required = True, help = "Path to a .pt state_dict saved by train.py")
    return parser.parse_args()

def load_model(checkpoint_path: str, config: SpikingTransformerConfig, device: torch.device) -> SpikingTransformer:
    """
    Instantiate a SpikingTransformer and load trained weights from disk. checkpoint_path is the path to a saved state_dict (as written by train.py).
    config is the config used to build the model (architecture must match the checkpoint). device is the device to load the model onto. Returns the 
    model in eval() mode, with weights loaded.
    """
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
    )

    state_dict = torch.load(checkpoint_path, map_location = device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model

def main() -> None:
    """
    Load a checkpoint, run it over the MNIST test set, and print accuracy.
    """
    args = parse_args()

    # Must match the architecture hyperparameters used in train.py's main(), only fields like batch_size are free to differ here.
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

    model = load_model(args.checkpoint, config, device)

    from train import evaluate as compute_accuracy
    from train import get_dataloaders

    _, test_loader = get_dataloaders(config)
    accuracy = compute_accuracy(model, test_loader, device)

    print(f"Test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
"""
Standalone evaluation script: load a trained checkpoint and report test accuracy.

To run this, enter the following command in terminal: "python evaluate.py --checkpoint path/to/checkpoint.pt" (defaults to MNIST; pass
--dataset cifar10/shd to match a checkpoint trained on a different dataset -- must match the dataset train.py was run with).
"""

import argparse

import torch

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
    Parse command-line arguments (at minimum, a --checkpoint path; --dataset selects the config preset the checkpoint was trained with).
    """
    parser = argparse.ArgumentParser(description = "Evaluate a trained SpikingTransformer checkpoint.")
    parser.add_argument("--checkpoint", type = str, required = True, help = "Path to a .pt state_dict saved by train.py")
    parser.add_argument("--dataset", type = str, default = "mnist", choices = sorted(_CONFIG_PRESETS), help = "Which dataset preset the checkpoint was trained with.")
    return parser.parse_args()

def load_model(checkpoint_path: str, config: SpikingTransformerConfig, device: torch.device) -> torch.nn.Module:
    """
    Instantiate a model and load trained weights from disk. checkpoint_path is the path to a saved state_dict (as written by train.py). config is
    the config used to build the model (architecture must match the checkpoint). device is the device to load the model onto. Returns the model
    in eval() mode, with weights loaded.
    """
    model = build_model(config)

    state_dict = torch.load(checkpoint_path, map_location = device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model

def main() -> None:
    """
    Load a checkpoint, run it over the test set, and print accuracy.
    """
    args = parse_args()

    # Must match the architecture hyperparameters train.py's build_config() used for this checkpoint's --dataset; fields like batch_size are
    # free to differ here.
    config = _CONFIG_PRESETS[args.dataset]()

    device = torch.device(config.device if (config.device != "mps" or torch.backends.mps.is_available()) else "cpu")

    model = load_model(args.checkpoint, config, device)

    from train import evaluate as compute_accuracy

    _, test_loader = get_dataloaders(config)
    accuracy = compute_accuracy(model, test_loader, device)

    print(f"Test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
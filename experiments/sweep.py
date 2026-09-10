"""
Ablation/benchmark sweep harness.

Ablation mode trains many small configurations of SpikingTransformer (varying num_timesteps, neuron_type, coding, and attention_mode, across
one or more datasets) on a small, fixed-size subset for speed, evaluates each, and logs accuracy/wall-clock/firing-rate/energy to a CSV under
experiments/results/. Headline mode instead does one full-dataset, full-epoch run per dataset (the SNN at its preset's default settings) plus one
matched ann_baseline_config run per dataset, logged to a separate CSV: these are the numbers that should be cited in docs/report.md.

To run this, enter one of the following commands in terminal:
  "python -m experiments.sweep" (ablation grid, MNIST only)
  "python -m experiments.sweep --datasets mnist cifar10 shd" (full ablation grid across all three datasets)
  "python -m experiments.sweep --headline" (headline runs, all three datasets, both model families)
  "python -m experiments.sweep --headline --datasets mnist" (headline runs, MNIST only)
"""

import argparse
import csv
import itertools
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset

from config import ann_baseline_config, cifar10_config, mnist_config, shd_config, SpikingTransformerConfig
from datasets import get_dataloaders
from train import evaluate, train_one_epoch
from transformer.energy import estimate_model_energy
from transformer.factory import build_model

_CONFIG_PRESETS = {"mnist": mnist_config, "cifar10": cifar10_config, "shd": shd_config}
RESULTS_DIR = Path(__file__).parent / "results"

# Ablation grid (only swept for the SNN; the ANN baseline has none of these knobs):
ABLATION_TIMESTEPS = [2, 4, 8, 16]
ABLATION_NEURON_TYPES = ["lif", "plif"]
ABLATION_CODINGS = ["repeat", "rate"] # "rate" is skipped for the "shd" dataset (config.py rejects it: SHD is already event-based).
ABLATION_ATTENTION_MODES = ["linear", "quadratic"]
ABLATION_EPOCHS = 3
ABLATION_TRAIN_SIZE = 2000
ABLATION_TEST_SIZE = 1000

CSV_FIELDS = [
    "dataset", "model_family", "num_timesteps", "neuron_type", "coding", "attention_mode",
    "test_accuracy", "wall_clock_seconds", "mean_firing_rate", "total_energy_pj", "total_dense_reference_energy_pj",
]

def _subset_loader(loader: DataLoader, size: int, shuffle: bool) -> DataLoader:
    """
    Build a DataLoader over the first min(size, len(dataset)) examples of loader's dataset, reusing loader's batch_size and collate_fn (needed
    for SHD's custom collate).
    """
    subset = Subset(loader.dataset, range(min(size, len(loader.dataset))))
    return DataLoader(subset, batch_size = loader.batch_size, shuffle = shuffle, collate_fn = loader.collate_fn)

def run_config(config: SpikingTransformerConfig, train_size: Optional[int] = None, test_size: Optional[int] = None) -> dict:
    """
    Build the model+data for config, train it for config.num_epochs, evaluate it, and (for an "snn" model_family) run one instrumented energy
    pass. train_size/test_size, if given, restrict training/evaluation to a subset (for fast ablation runs); leave both None for a full,
    "headline" run. Returns a dict of result fields matching CSV_FIELDS (minus the sweep-parameter columns, which the caller fills in).
    """
    device = torch.device("cpu")
    train_loader, test_loader = get_dataloaders(config)

    if train_size is not None:
        train_loader = _subset_loader(train_loader, train_size, shuffle = True)
    if test_size is not None:
        test_loader = _subset_loader(test_loader, test_size, shuffle = False)

    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)

    start = time.perf_counter()
    for _ in range(config.num_epochs):
        train_one_epoch(model, train_loader, optimizer, device)
    wall_clock_seconds = time.perf_counter() - start

    test_accuracy = evaluate(model, test_loader, device)

    sample_input, _ = next(iter(test_loader))
    sample_input = sample_input[:1].to(device)
    energy_timesteps = config.num_timesteps if config.model_family == "snn" else 1
    report = estimate_model_energy(model, sample_input, energy_timesteps)

    ac_records = [r for r in report.layers if r.op_type == "AC"]
    mean_firing_rate = sum(r.mean_firing_rate for r in ac_records) / len(ac_records) if ac_records else 0.0

    return {
        "test_accuracy": test_accuracy,
        "wall_clock_seconds": wall_clock_seconds,
        "mean_firing_rate": mean_firing_rate,
        "total_energy_pj": report.total_energy_pj,
        "total_dense_reference_energy_pj": report.total_dense_reference_energy_pj,
    }

def iter_ablation_configs(datasets: list):
    """
    Yield (config, sweep_params) pairs for the full ablation grid over the given datasets, where sweep_params is a dict recording which grid
    point this config corresponds to (for CSV logging).
    """
    for dataset in datasets:
        preset = _CONFIG_PRESETS[dataset]
        codings = ["repeat"] if dataset == "shd" else ABLATION_CODINGS

        for num_timesteps, neuron_type, coding, attention_mode in itertools.product(
            ABLATION_TIMESTEPS, ABLATION_NEURON_TYPES, codings, ABLATION_ATTENTION_MODES
        ):
            config = preset(
                num_timesteps = num_timesteps,
                neuron_type = neuron_type,
                coding = coding,
                attention_mode = attention_mode,
                num_epochs = ABLATION_EPOCHS,
            )
            sweep_params = {
                "dataset": dataset,
                "model_family": "snn",
                "num_timesteps": num_timesteps,
                "neuron_type": neuron_type,
                "coding": coding,
                "attention_mode": attention_mode,
            }
            yield config, sweep_params

def run_ablation(datasets: list) -> Path:
    """
    Run the full ablation grid over datasets and write one CSV row per configuration. Returns the path of the written CSV.
    """
    RESULTS_DIR.mkdir(parents = True, exist_ok = True)
    csv_path = RESULTS_DIR / f"sweep_{int(time.time())}.csv"

    configs = list(iter_ablation_configs(datasets))
    print(f"Running {len(configs)} ablation configurations across datasets: {datasets}")

    with open(csv_path, "w", newline = "") as f:
        writer = csv.DictWriter(f, fieldnames = CSV_FIELDS)
        writer.writeheader()

        for i, (config, sweep_params) in enumerate(configs):
            print(f"[{i + 1}/{len(configs)}] {sweep_params}")
            result = run_config(config, train_size = ABLATION_TRAIN_SIZE, test_size = ABLATION_TEST_SIZE)
            writer.writerow({**sweep_params, **result})
            f.flush()

    print(f"Wrote ablation results to {csv_path}")
    return csv_path

def run_headline(datasets: list) -> Path:
    """
    Run one full-dataset, full-epoch SNN run and one matched ANN baseline run per dataset, and write one CSV row per run. Returns the path of
    the written CSV.
    """
    RESULTS_DIR.mkdir(parents = True, exist_ok = True)
    csv_path = RESULTS_DIR / f"headline_{int(time.time())}.csv"

    with open(csv_path, "w", newline = "") as f:
        writer = csv.DictWriter(f, fieldnames = CSV_FIELDS)
        writer.writeheader()

        for dataset in datasets:
            preset = _CONFIG_PRESETS[dataset]

            snn_config = preset()
            print(f"[headline] {dataset} snn: {snn_config.num_epochs} epochs, full dataset")
            snn_result = run_config(snn_config)
            writer.writerow({
                "dataset": dataset, "model_family": "snn", "num_timesteps": snn_config.num_timesteps,
                "neuron_type": snn_config.neuron_type, "coding": snn_config.coding, "attention_mode": snn_config.attention_mode,
                **snn_result,
            })
            f.flush()

            ann_config = ann_baseline_config(dataset)
            print(f"[headline] {dataset} ann: {ann_config.num_epochs} epochs, full dataset")
            ann_result = run_config(ann_config)
            writer.writerow({
                "dataset": dataset, "model_family": "ann", "num_timesteps": None,
                "neuron_type": None, "coding": None, "attention_mode": None,
                **ann_result,
            })
            f.flush()

    print(f"Wrote headline results to {csv_path}")
    return csv_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Run the SpikingTransformer ablation/benchmark sweep.")
    parser.add_argument("--headline", action = "store_true", help = "Run headline (full-dataset) runs instead of the ablation grid.")
    parser.add_argument("--datasets", nargs = "+", default = ["mnist"], choices = sorted(_CONFIG_PRESETS), help = "Which datasets to run.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if args.headline:
        run_headline(args.datasets)
    else:
        run_ablation(args.datasets)

if __name__ == "__main__":
    main()
"""
Plotting utilities for experiments/sweep.py's output CSVs.

Produces three figures under experiments/results/figures/:
  (a) accuracy vs. num_timesteps, one line per (neuron_type, coding) combination: from an ablation-mode CSV.
  (b) an accuracy-vs-energy Pareto scatter: every SNN ablation point, plus the ANN baseline point(s) from a headline-mode CSV, highlighted.
  (c) a per-layer firing-rate bar chart and a spike raster for one representative trained model.

To run this, enter the following command in terminal:
  "python -m experiments.plotting --sweep-csv experiments/results/sweep_<timestamp>.csv --headline-csv experiments/results/headline_<timestamp>.csv"
"""

import argparse
import csv
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

FIGURES_DIR = Path(__file__).parent / "results" / "figures"

def _read_csv(path: Path) -> list:
    """
    Read a sweep/headline CSV into a list of dicts, converting numeric fields back from strings (empty strings, from the ANN baseline's
    SNN-only fields, become None).
    """
    numeric_fields = {"num_timesteps", "test_accuracy", "wall_clock_seconds", "mean_firing_rate", "total_energy_pj", "total_dense_reference_energy_pj"}
    rows = []

    with open(path, newline = "") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for key, value in row.items():
                if value == "":
                    parsed[key] = None
                elif key == "num_timesteps":
                    parsed[key] = int(value)
                elif key in numeric_fields:
                    parsed[key] = float(value)
                else:
                    parsed[key] = value
            rows.append(parsed)

    return rows

def plot_accuracy_vs_timesteps(sweep_rows: list, output_path: Path) -> None:
    """
    Plot test_accuracy vs. num_timesteps, one line per (neuron_type, coding) combination present in sweep_rows, averaged over any other swept
    fields (e.g. attention_mode, dataset) at each num_timesteps value.
    """
    groups = {}
    for row in sweep_rows:
        key = (row["neuron_type"], row["coding"])
        groups.setdefault(key, {}).setdefault(row["num_timesteps"], []).append(row["test_accuracy"])

    fig, ax = plt.subplots(figsize = (7, 5))
    for (neuron_type, coding), by_timestep in sorted(groups.items()):
        timesteps = sorted(by_timestep)
        accuracies = [sum(by_timestep[t]) / len(by_timestep[t]) for t in timesteps]
        ax.plot(timesteps, accuracies, marker = "o", label = f"{neuron_type}, {coding}")

    ax.set_xlabel("num_timesteps (T)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy vs. simulation timesteps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

def plot_energy_pareto(sweep_rows: list, headline_rows: Optional[list], output_path: Path) -> None:
    """
    Plot an accuracy-vs-energy Pareto scatter: every SNN row from sweep_rows (colored by neuron_type), plus every ANN row from headline_rows (if
    given), highlighted distinctly.
    """
    fig, ax = plt.subplots(figsize = (7, 5))

    for neuron_type, marker in [("lif", "o"), ("plif", "^")]:
        points = [(r["total_energy_pj"], r["test_accuracy"]) for r in sweep_rows if r["neuron_type"] == neuron_type]
        if points:
            xs, ys = zip(*points)
            ax.scatter(xs, ys, marker = marker, alpha = 0.6, label = f"SNN ({neuron_type})")

    if headline_rows:
        ann_points = [(r["total_energy_pj"], r["test_accuracy"]) for r in headline_rows if r["model_family"] == "ann"]
        if ann_points:
            xs, ys = zip(*ann_points)
            ax.scatter(xs, ys, marker = "*", s = 200, color = "red", label = "ANN baseline", zorder = 5)

    ax.set_xscale("log")
    ax.set_xlabel("estimated energy per sample (pJ, log scale)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy vs. estimated energy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

def plot_firing_rates(model: torch.nn.Module, sample_input: torch.Tensor, num_timesteps: int, output_path: Path) -> None:
    """
    Run one instrumented forward pass of model on sample_input and plot a bar chart of each LIF neuron's mean firing rate, ordered by depth.
    """
    from transformer.energy import estimate_model_energy

    report = estimate_model_energy(model, sample_input, num_timesteps)
    ac_records = [r for r in report.layers if r.op_type == "AC"]

    fig, ax = plt.subplots(figsize = (9, 5))
    names = [r.name for r in ac_records]
    rates = [r.mean_firing_rate for r in ac_records]
    ax.barh(names, rates, color = "steelblue")
    ax.set_xlabel("mean firing rate")
    ax.set_title("Per-layer mean firing rate (AC-classified operations)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Plot figures from experiments/sweep.py's output CSVs.")
    parser.add_argument("--sweep-csv", type = str, required = True, help = "Path to an ablation-mode CSV from experiments/sweep.py.")
    parser.add_argument("--headline-csv", type = str, default = None, help = "Path to a headline-mode CSV from experiments/sweep.py (optional).")
    parser.add_argument("--checkpoint", type = str, default = None, help = "Path to a trained SNN checkpoint (from train.py), for the firing-rate plot.")
    parser.add_argument("--dataset", type = str, default = "mnist", choices = ["mnist", "cifar10", "shd"], help = "Dataset the checkpoint was trained on.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    FIGURES_DIR.mkdir(parents = True, exist_ok = True)

    sweep_rows = _read_csv(Path(args.sweep_csv))
    headline_rows = _read_csv(Path(args.headline_csv)) if args.headline_csv else None

    plot_accuracy_vs_timesteps(sweep_rows, FIGURES_DIR / "accuracy_vs_timesteps.png")
    print(f"Wrote {FIGURES_DIR / 'accuracy_vs_timesteps.png'}")

    plot_energy_pareto(sweep_rows, headline_rows, FIGURES_DIR / "energy_pareto.png")
    print(f"Wrote {FIGURES_DIR / 'energy_pareto.png'}")

    if args.checkpoint:
        import torch as _torch

        from config import cifar10_config, mnist_config, shd_config
        from datasets import get_dataloaders
        from evaluate import load_model

        preset = {"mnist": mnist_config, "cifar10": cifar10_config, "shd": shd_config}[args.dataset]
        config = preset()
        device = _torch.device("cpu")
        model = load_model(args.checkpoint, config, device)

        _, test_loader = get_dataloaders(config)
        sample_input, _ = next(iter(test_loader))
        sample_input = sample_input[:1].to(device)

        plot_firing_rates(model, sample_input, config.num_timesteps, FIGURES_DIR / "firing_rates.png")
        print(f"Wrote {FIGURES_DIR / 'firing_rates.png'}")
    else:
        print("No --checkpoint given; skipping the firing-rate plot (pass --checkpoint path/to/checkpoint.pt --dataset <name> to include it).")

if __name__ == "__main__":
    main()
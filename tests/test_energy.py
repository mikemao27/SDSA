"""
Correctness tests for transformer/energy.py's SOP/energy accounting, against hand-computed toy examples.
"""

import torch
import torch.nn as nn

from transformer.energy import ENERGY_PER_AC_PJ, ENERGY_PER_MAC_PJ, estimate_model_energy

def test_binary_input_is_classified_ac() -> None:
    """
    A bare nn.Linear(4, 8, bias = False) fed a binary input with exactly 50% of elements firing should be classified as an AC (spike-driven) op,
    with SOPs discounted by that 50% firing rate: total_sops == (4 * 8) * 1 application * 0.5, energy_pj == total_sops * ENERGY_PER_AC_PJ.
    """
    model = nn.Linear(4, 8, bias = False)
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]) # (batch = 2, in_features = 4), exactly 50% ones.

    report = estimate_model_energy(model, x, num_timesteps = 1)

    assert len(report.layers) == 1
    record = report.layers[0]
    assert record.op_type == "AC"
    expected_sops = (4 * 8) * 1 * 0.5
    assert record.total_sops == expected_sops
    assert record.energy_pj == expected_sops * ENERGY_PER_AC_PJ
    assert report.total_energy_pj == record.energy_pj

def test_continuous_input_is_classified_mac() -> None:
    """
    The same layer fed a continuous (non-binary) input should be classified as a MAC op, undiscounted (mean_firing_rate == 1.0):
    total_sops == (4 * 8) * 1 application * 1.0, energy_pj == total_sops * ENERGY_PER_MAC_PJ.
    """
    torch.manual_seed(0)
    model = nn.Linear(4, 8, bias = False)
    x = torch.randn(2, 4)

    report = estimate_model_energy(model, x, num_timesteps=1)

    assert len(report.layers) == 1
    record = report.layers[0]
    assert record.op_type == "MAC"
    assert record.mean_firing_rate == 1.0
    expected_sops = (4 * 8) * 1 * 1.0
    assert record.total_sops == expected_sops
    assert record.energy_pj == expected_sops * ENERGY_PER_MAC_PJ

def test_end_to_end_model_energy_runs() -> None:
    """
    estimate_model_energy should run cleanly on a full SpikingTransformer and produce a report with both AC and MAC layers, whose total_energy_pj
    is strictly positive and whose total_dense_reference_energy_pj is also strictly positive.
    """
    from transformer.model import SpikingTransformer

    torch.manual_seed(0)
    model = SpikingTransformer(image_size = 28, patch_size = 7, in_channels = 1, num_classes = 10, embed_dim = 16, depth = 2, num_heads = 2, num_timesteps = 3)
    sample_input = torch.rand(1, 1, 28, 28)

    report = estimate_model_energy(model, sample_input, num_timesteps=3)

    op_types = {record.op_type for record in report.layers}
    assert op_types == {"AC", "MAC"}
    assert report.total_energy_pj > 0.0
    assert report.total_dense_reference_energy_pj > 0.0

    # track_firing_rate must be restored to its original (False) setting on every LIF neuron after the call.
    from transformer.neurons import LIFNeuron
    assert all(not m.track_firing_rate for m in model.modules() if isinstance(m, LIFNeuron))
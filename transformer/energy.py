"""
Theoretical energy accounting for spiking vs. dense models.

This module estimates, for a single forward pass over one sample, how much energy the model's linear-algebra operations would consume under the
standard accounting used in the spiking-neural-network literature (e.g. Spikformer, Spike-Driven Transformer): every operation is classified as
either an accumulate (AC, a spike-triggered addition: cheap) or a multiply-accumulate (MAC, an ordinary dense floating-point op: expensive),
each op type has a fixed per-operation energy cost, and AC ops are further discounted by how sparse the driving spike tensor actually is (its mean
firing rate). This is a standard, widely-cited simplification, not a cycle-accurate hardware simulation: see the classification rule below and
the caveats in docs/report.md.

Energy constants (45nm CMOS, Horowitz, "Computing's Energy Problem (and what we can do about it)", ISSCC 2014), these exact figures are the ones
reused throughout the spiking-transformer literature (Zhou et al., "Spikformer"; Yao et al., "Spike-Driven Transformer"):
  - ENERGY_PER_AC_PJ: energy of one 32-bit floating-point ADD, i.e. what a spike "costs" to propagate.
  - ENERGY_PER_MAC_PJ: energy of one 32-bit floating-point MULTIPLY-ADD, i.e. what a dense (non-spiking) op costs.

Classification rule: an operation is AC (addition-only) if at least one of its operands is a binary ({0, 1}-valued) spike tensor, multiplying
anything by a binary value reduces to a conditional add, never a real multiply. Its synaptic-operation (SOP) count is discounted by that operand's
mean firing rate (if both operands are binary, the convention is to use the left operand's rate). An operation with no binary operand is a MAC,
counted at its full (undiscounted) rate of 1.0.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import torch
import torch.nn as nn

from .attention import SpikingSelfAttention
from .neurons import LIFNeuron

ENERGY_PER_AC_PJ = 0.9
ENERGY_PER_MAC_PJ = 4.6

@dataclass
class LayerEnergyRecord:
    """
    Energy accounting for one operation (an nn.Linear/nn.Conv2d layer, or one of SpikingSelfAttention's two internal matmuls) during a single
    instrumented forward pass. name identifies the operation (its module path, or a matmul label for attention). op_type is "AC" or "MAC".
    macs_per_application is the multiply-accumulate cost of a single application of this operation (e.g. in_features * out_features for a Linear).
    num_applications is how many times that operation was applied per sample (e.g. across timesteps and tokens). mean_firing_rate is the driving
    operand's mean firing rate (1.0 for MAC ops, which are not discounted). total_sops is macs_per_application * num_applications *
    mean_firing_rate. energy_pj is total_sops times the energy-per-op constant for this op_type.
    """
    name: str
    op_type: str
    macs_per_application: float
    num_applications: float
    mean_firing_rate: float
    total_sops: float
    energy_pj: float

@dataclass
class EnergyReport:
    """
    The full per-sample energy estimate for one instrumented forward pass. layers holds one LayerEnergyRecord per accounted operation.
    total_energy_pj is the sum of every record's energy_pj: the model's actual estimated per-sample energy, given its measured sparsity.
    total_dense_reference_energy_pj is a sanity-check reference number: what the same architecture would cost if every operation were a full-rate
    MAC (i.e. dense, non-spiking) and evaluated once (T = 1) instead of num_timesteps times: an upper-bound baseline to compare the real,
    sparsity-discounted total_energy_pj against.
    """
    layers: list[LayerEnergyRecord] = field(default_factory=list)
    total_energy_pj: float = 0.0
    total_dense_reference_energy_pj: float = 0.0

def _is_binary(x: torch.Tensor) -> bool:
    """
    Return True if every element of x is exactly 0.0 or 1.0 (i.e. x is a true spike tensor).
    """
    return bool(torch.all((x == 0.0) | (x == 1.0)).item())

def _record_from_counts(name: str, macs_per_application: float, num_applications: float, is_binary_operand: bool, firing_rate: float) -> LayerEnergyRecord:
    op_type = "AC" if is_binary_operand else "MAC"
    mean_firing_rate = firing_rate if is_binary_operand else 1.0
    total_sops = macs_per_application * num_applications * mean_firing_rate
    energy_per_op = ENERGY_PER_AC_PJ if op_type == "AC" else ENERGY_PER_MAC_PJ
    
    return LayerEnergyRecord(
        name = name,
        op_type = op_type,
        macs_per_application = macs_per_application,
        num_applications = num_applications,
        mean_firing_rate = mean_firing_rate,
        total_sops = total_sops,
        energy_pj = total_sops * energy_per_op,
    )

def _linear_hook(module: nn.Linear, inputs: tuple, output: torch.Tensor, records: list, name: str, batch_size: int) -> None:
    """
    Forward hook for nn.Linear: classifies AC/MAC from the actual input tensor and records one LayerEnergyRecord.

    batch_size is the true batch size of the top-level sample_input passed to estimate_model_energy: NOT inferred from x.shape[0], since this
    layer's actual input may be shaped (T, B, N, in_features) (batch on axis 1, not axis 0) rather than a plain (B, in_features).
    """
    x = inputs[0]
    num_applications = x.numel() // x.shape[-1] // batch_size
    is_binary_operand = _is_binary(x)
    firing_rate = x.float().mean().item() if is_binary_operand else 1.0
    macs_per_application = float(module.in_features * module.out_features)

    records.append(_record_from_counts(name, macs_per_application, num_applications, is_binary_operand, firing_rate))

def _conv2d_hook(module: nn.Conv2d, inputs: tuple, output: torch.Tensor, records: list, name: str) -> None:
    """
    Forward hook for nn.Conv2d: classifies AC/MAC from the actual input tensor and records one LayerEnergyRecord.
    """
    x = inputs[0]
    out_h, out_w = output.shape[-2], output.shape[-1]
    num_applications = float(out_h * out_w)
    kernel_h, kernel_w = module.kernel_size
    macs_per_application = float((module.in_channels // module.groups) * module.out_channels * kernel_h * kernel_w)
    is_binary_operand = _is_binary(x)
    firing_rate = x.float().mean().item() if is_binary_operand else 1.0

    records.append(_record_from_counts(name, macs_per_application, num_applications, is_binary_operand, firing_rate))

def _firing_rate(neuron: LIFNeuron, label: str) -> float:
    if neuron.last_firing_rate is None:
        raise RuntimeError(f"{label} has no last_firing_rate; estimate_model_energy() must run the forward pass with firing-rate tracking enabled.")
    return neuron.last_firing_rate.item()

def _attention_hook(module: SpikingSelfAttention, inputs: tuple, output: torch.Tensor, records: list, name: str) -> None:
    """
    Forward hook for SpikingSelfAttention: analytically accounts for its two internal matmuls (which aren't nn.Modules and so aren't caught by
    the generic Linear/Conv2d hooks above), using the firing rates its own q_lif/k_lif/v_lif already recorded during this same forward pass.
    """
    T, B, N, C = inputs[0].shape
    H = module.num_heads
    d = C // H

    q_rate = _firing_rate(module.q_lif, f"{name}.q_lif")
    k_rate = _firing_rate(module.k_lif, f"{name}.k_lif")
    v_rate = _firing_rate(module.v_lif, f"{name}.v_lif")

    if module.attention_mode == "quadratic":
        # Q @ K^T: (N, d) @ (d, N) -> (N, N); both operands binary, convention: use the left operand's (Q's) rate.
        records.append(_record_from_counts(f"{name}.q_kT", float(N * N * d), float(T * H), True, q_rate))
        # attn @ V: (N, N) @ (N, d) -> (N, d); attn is not binary, V is -> use V's rate.
        records.append(_record_from_counts(f"{name}.attn_v", float(N * N * d), float(T * H), True, v_rate))
    else:
        # K^T @ V: (d, N) @ (N, d) -> (d, d); both operands binary, convention: use the left operand's (K's) rate.
        records.append(_record_from_counts(f"{name}.kT_v", float(d * d * N), float(T * H), True, k_rate))
        # Q @ (K^T V): (N, d) @ (d, d) -> (N, d); Q is binary, K^T V is not -> use Q's rate.
        records.append(_record_from_counts(f"{name}.q_kv", float(N * d * d), float(T * H), True, q_rate))

@contextmanager
def _firing_rate_tracking(model: nn.Module) -> Iterator[None]:
    """
    Temporarily set track_firing_rate = True on every LIFNeuron in model, restoring each neuron's original setting on exit.
    """
    lif_modules = [m for m in model.modules() if isinstance(m, LIFNeuron)]
    previous_settings = [m.track_firing_rate for m in lif_modules]

    for m in lif_modules:
        m.track_firing_rate = True

    try:
        yield
    finally:
        for m, was_tracking in zip(lif_modules, previous_settings):
            m.track_firing_rate = was_tracking

def estimate_model_energy(model: nn.Module, sample_input: torch.Tensor, num_timesteps: int) -> EnergyReport:
    """
    Run one instrumented forward pass of model on sample_input and return an EnergyReport estimating its per-sample energy consumption.

    model is put into eval() mode; every nn.Linear and nn.Conv2d submodule is hooked for generic AC/MAC classification, and every
    SpikingSelfAttention submodule is separately hooked to account for its two internal matmuls. All LIF neurons in model have firing-rate tracking
    temporarily enabled for the duration of this call (and restored to their previous setting afterward), so this is safe to call on a model that
    was trained with track_firing_rate = False. num_timesteps is the model's T (used only to compute total_dense_reference_energy_pj, a "what would
    a single dense timestep of this same architecture cost" reference number: every per-sample count derived from real tensor shapes above
    already scales with T on its own).
    """
    model.eval()
    records: list[LayerEnergyRecord] = []
    hooks = []
    batch_size = sample_input.shape[0]

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(functools.partial(_linear_hook, records = records, name = name, batch_size = batch_size)))
        elif isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(functools.partial(_conv2d_hook, records = records, name = name)))
        elif isinstance(module, SpikingSelfAttention):
            hooks.append(module.register_forward_hook(functools.partial(_attention_hook, records = records, name = name)))

    try:
        with _firing_rate_tracking(model), torch.no_grad():
            model(sample_input)
    finally:
        for hook in hooks:
            hook.remove()

    total_energy_pj = sum(r.energy_pj for r in records)
    total_dense_macs = sum(r.macs_per_application * r.num_applications for r in records)
    total_dense_reference_energy_pj = (total_dense_macs / num_timesteps) * ENERGY_PER_MAC_PJ

    return EnergyReport(layers = records, total_energy_pj = total_energy_pj, total_dense_reference_energy_pj = total_dense_reference_energy_pj)
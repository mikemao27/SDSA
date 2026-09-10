"""
Leaky Integrate-and-Fire (LIF) neuron module.

A LIF neuron integrates incoming current into a membrane potential that leaks (decays) toward zero over time, and emits a spike whenever the
potential crosses a threshold, after which the potential is reset. This is the basic computational unit that stands in for the usual continuous-valued
activations (e.g. GELU, softmax) inside the transformer blocks, and it is also what turns the Q/K/V projections into true binary spike trains.
"""

import math

import torch
import torch.nn as nn

from .surrogate import spike_function

class LIFNeuron(nn.Module):
    """
    A layer of leaky integrate-and-fire neurons, simulated over a time dimension.

    Given a sequence of input currents of shape (T, ...), this module maintains a membrane potential across time-steps, applies leak/decay, and
    emits a binary spike (via the surrogate-gradient spike_function) whenever the potential crosses threshold. The membrane potential is reduced after
    each spike according to reset_mechanism.

    The threshold is the firing threshold for the membrane potential. beta is the leak/decay factor applied to the membrane potential at each time-step
    (0 < beta < 1; closer to 1 = slower leak). alpha is the surrogate gradient steepness, passed through to spike_function. reset_mechanism is either
    "subtract" (subtract threshold from the membrane potential on spike) or "zero" (hard-reset the membrane potential to 0 on spike). learnable if True,
    means that threshold and beta are registered as learnable parameters instead of fixed constants. track_firing_rate if True, records the mean
    firing rate of the neuron's most recent forward() call in self.last_firing_rate (used by transformer/energy.py's energy accounting; left False
    by default so ordinary training pays no extra cost).
    """
    def __init__(
        self,
        threshold: float = 1.0,
        beta: float = 0.9,
        alpha: float = 2.0,
        reset_mechanism: str = "subtract",
        learnable: bool = False,
        track_firing_rate: bool = False,
    ) -> None:
        if reset_mechanism not in {"subtract", "zero"}:
            raise ValueError("reset_mechanism must be either 'subtract' or 'zero'.")

        super().__init__()
        self.alpha = alpha
        self.reset_mechanism = reset_mechanism
        self.track_firing_rate = track_firing_rate
        self.last_firing_rate: torch.Tensor | None = None

        if learnable:
            self.threshold = nn.Parameter(torch.tensor(threshold))
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer("threshold", torch.tensor(threshold))
            self.register_buffer("beta", torch.tensor(beta))

    def _decay(self) -> torch.Tensor:
        """
        Return the per-time-step decay factor applied to the membrane potential. Subclasses (e.g. PLIFNeuron) override this to substitute a
        learnable, reparameterized decay; plain LIFNeuron just returns self.beta unchanged.
        """
        return self.beta

    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        """
        Simulate the neuron over the time dimension of input_current. Membrane potential starts at 0 at the first time-step of every call (i.e. the
        state is not carried over between separate forward() calls). input_current is a tensor of shape (T, B, *feature_dims) giving the input current
        injected into the neuron at each of T time-steps. Returns a tensor of shape (T, B, *feature_dims) containing binary spikes (0.0 / 1.0) emitted
        at each time-step.
        """
        membrane_potential = torch.zeros_like(input_current[0])
        spikes = []
        decay = self._decay()

        for timestep in range(input_current.shape[0]):
            membrane_potential = decay * membrane_potential + input_current[timestep]
            spike = spike_function(membrane_potential, self.threshold, self.alpha)

            if self.reset_mechanism == "subtract":
                membrane_potential = membrane_potential - spike * self.threshold
            else:
                membrane_potential = membrane_potential * (1.0 - spike)

            spikes.append(spike)

        spikes = torch.stack(spikes, dim = 0)

        if self.track_firing_rate:
            self.last_firing_rate = spikes.detach().mean()

        return spikes

class PLIFNeuron(LIFNeuron):
    """
    Parametric LIF (PLIF) neuron: a LIFNeuron whose decay is a learnable parameter, reparameterized through a sigmoid so it always stays in the
    valid (0, 1) range under gradient steps (Fang et al., "Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural
    Networks", 2021). This fixes a real gap in plain LIFNeuron's own `learnable=True` path, where beta is a bare nn.Parameter that gradient steps
    can push outside (0, 1) with no constraint.

    Concretely, decay = sigmoid(w), where w is an nn.Parameter initialized to the inverse-sigmoid of init_beta (so the neuron starts at exactly
    init_beta's decay rate). If num_channels is given, w (and therefore decay) is a per-channel vector of that width instead of a shared scalar,
    letting different feature channels learn different time constants: it broadcasts against the trailing feature dimension of whatever tensor
    this neuron is applied to. threshold is also made learnable (matching the intent of LIFNeuron's `learnable = True` flag). All other
    arguments (threshold's initial value, alpha, reset_mechanism, track_firing_rate) behave exactly as in LIFNeuron.
    """
    def __init__(
        self,
        threshold: float = 1.0,
        init_beta: float = 0.9,
        alpha: float = 2.0,
        reset_mechanism: str = "subtract",
        num_channels: int | None = None,
        track_firing_rate: bool = False,
    ) -> None:
        super().__init__(
            threshold = threshold,
            beta = init_beta,
            alpha = alpha,
            reset_mechanism = reset_mechanism,
            learnable = False,
            track_firing_rate = track_firing_rate,
        )

        init_w = math.log(init_beta / (1.0 - init_beta))
        shape = (num_channels,) if num_channels is not None else ()
        self.w = nn.Parameter(torch.full(shape, init_w))
        self.threshold = nn.Parameter(torch.tensor(threshold))

    def _decay(self) -> torch.Tensor:
        return torch.sigmoid(self.w)

def build_neuron(
    neuron_type: str,
    *,
    threshold: float = 1.0,
    beta: float = 0.9,
    alpha: float = 2.0,
    reset_mechanism: str = "subtract",
    channels: int | None = None,
    track_firing_rate: bool = False,
) -> LIFNeuron:
    """
    Factory for constructing a spiking neuron layer by name, used by every call site in the model (SpikingSelfAttention, SpikingMLP) instead of
    constructing LIFNeuron directly. neuron_type is "lif" for a plain LIFNeuron (channels is ignored, decay/threshold are shared scalars) or "plif"
    for a PLIFNeuron (channels, if given, sizes its per-channel learnable decay). channels, if given, is the feature width the neuron will be
    applied to. track_firing_rate is forwarded to the constructed neuron. Raises ValueError for an unrecognized neuron_type.
    """
    if neuron_type == "lif":
        return LIFNeuron(
            threshold = threshold,
            beta = beta,
            alpha = alpha,
            reset_mechanism = reset_mechanism,
            track_firing_rate = track_firing_rate,
        )

    if neuron_type == "plif":
        return PLIFNeuron(
            threshold = threshold,
            init_beta = beta,
            alpha = alpha,
            reset_mechanism = reset_mechanism,
            num_channels = channels,
            track_firing_rate = track_firing_rate,
        )

    raise ValueError(f"Unrecognized neuron_type: {neuron_type!r}. Supported types: 'lif', 'plif'.")
"""
Leaky Integrate-and-Fire (LIF) neuron module.

A LIF neuron integrates incoming current into a membrane potential that leaks (decays) toward zero over time, and emits a spike whenever the
potential crosses a threshold, after which the potential is reset. This is the basic computational unit that stands in for the usual continuous-valued
activations (e.g. GELU, softmax) inside the transformer blocks, and it is also what turns the Q/K/V projections into true binary spike trains.
"""

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
    means that threshold and beta are registered as learnable parameters instead of fixed constants.
    """
    def __init__(
        self,
        threshold: float = 1.0,
        beta: float = 0.9,
        alpha: float = 2.0,
        reset_mechanism: str = "subtract",
        learnable: bool = False,
    ) -> None:
        if reset_mechanism not in {"subtract", "zero"}:
            raise ValueError("reset_mechanism must be either 'subtract' or 'zero'.")

        super().__init__()
        self.alpha = alpha
        self.reset_mechanism = reset_mechanism

        if learnable:
            self.threshold = nn.Parameter(torch.tensor(threshold))
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer("threshold", torch.tensor(threshold))
            self.register_buffer("beta", torch.tensor(beta))

    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        """
        Simulate the neuron over the time dimension of input_current. Membrane potential starts at 0 at the first time-step of every call (i.e. the 
        state is not carried over between separate forward() calls). input_current is a tensor of shape (T, B, *feature_dims) giving the input current 
        injected into the neuron at each of T time-steps. Returns a tensor of shape (T, B, *feature_dims) containing binary spikes (0.0 / 1.0) emitted 
        at each time-step.
        """
        membrane_potential = torch.zeros_like(input_current[0])
        spikes = []

        for timestep in range(input_current.shape[0]):
            membrane_potential = self.beta * membrane_potential + input_current[timestep]
            spike = spike_function(membrane_potential, self.threshold, self.alpha)

            if self.reset_mechanism == "subtract":
                membrane_potential = membrane_potential - spike * self.threshold
            else:
                membrane_potential = membrane_potential * (1.0 - spike)
            
            spikes.append(spike)
        
        return torch.stack(spikes, dim = 0)
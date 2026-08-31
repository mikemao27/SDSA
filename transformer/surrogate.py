"""
Surrogate gradient function for the non-differentiable spiking nonlinearity.

The forward pass of a spiking neuron applies a hard threshold (heaviside step function) to produce a binary spike (0.0 or 1.0). This function has zero
gradient almost everywhere, which makes it unusable with standard back-propagation. The common workaround in surrogate-gradient learning is to
keep the exact Heaviside step in the forward pass, but substitute the derivative of a smooth, well-behaved function in the backward pass.

This module defines that forward/backward pair as a torch.autograd.Function.
"""

import torch

class FastSigmoidSpike(torch.autograd.Function):
    """
    Heaviside step forward pass with a fast-sigmoid surrogate gradient backward pass. In the forward, spike = 1.0 if 
    (membrane_potential - threshold) >= 0 else 0.0. Backward (surrogate), we substitute the derivative of a fast sigmoid, e.g. 
    d/dx [x / (1 + alpha * |x|)] = 1 / (alpha * |x| + 1) ** 2 evaluated at (membrane_potential - threshold), in place of the true (zero almost 
    everywhere) derivative of the heaviside step.
    """
    @staticmethod
    def forward(context, membrane_potential: torch.Tensor, threshold: float, alpha: float) -> torch.Tensor:
        """
        Compute the binary spike tensor and stash values needed for backward(). context is the autograd context for saving tensors/values between 
        forward and backward. membrane_potential is a tensor of membrane potentials, any shape. threshold is the firing threshold applied element-wise.
        alpha is the surrogate gradient steepness parameter.

        Returns a tensor of the same shape as membrane_potential, containing only 0.0 and 1.0 values.
        """
        shifted = membrane_potential - threshold
        context.save_for_backward(shifted)
        context.alpha = alpha
        spike = (shifted >= 0).to(membrane_potential.dtype)
        return spike

    @staticmethod
    def backward(context, gradient_output: torch.Tensor):
        """
        Compute the surrogate gradient w.r.t. membrane_potential. context is the autograd context populated in forward(). gradient_output is the 
        upstream gradient with respect to this function's output. Returns a 3-tuple of gradients matching forward()'s inputs: 
        (grad_membrane_potential, None, None) since threshold and alpha are plain floats, not learnable tensors.
        """
        (shifted,) = context.saved_tensors
        alpha = context.alpha
        surrogate_gradient = 1.0 / (alpha * shifted.abs() + 1.0) ** 2
        gradient_input = gradient_output * surrogate_gradient
        return gradient_input, None, None


def spike_function(membrane_potential: torch.Tensor, threshold: float = 1.0, alpha: float = 2.0) -> torch.Tensor:
    """
    Convenience wrapper around FastSigmoidSpike.apply. membrane_potential is a tensor of membrane potentials. threshold is the firing threshold.
    alpha is the surrogate gradient steepness. Returns a binary spike tensor (0.0 / 1.0), same shape as membrane_potential.
    """
    return FastSigmoidSpike.apply(membrane_potential, threshold, alpha)
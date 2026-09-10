"""
Correctness tests for PLIFNeuron: its decay must stay in the valid (0, 1) range under gradient steps (unlike LIFNeuron's bare-Parameter
`learnable = True` path), and its per-channel decay must broadcast correctly.
"""

import torch

from transformer.neurons import LIFNeuron, PLIFNeuron, build_neuron

def test_decay_stays_in_unit_interval_under_aggressive_training() -> None:
    """
    Run many optimizer steps with an aggressive learning rate on a loss that pushes w toward extremes; sigmoid(w) (the actual decay used) must
    remain strictly inside (0, 1) throughout, even though w itself is unconstrained.
    """
    torch.manual_seed(0)
    neuron = PLIFNeuron(init_beta = 0.9, num_channels = 4)
    optimizer = torch.optim.SGD(neuron.parameters(), lr = 1.0)

    x = torch.rand(6, 2, 4) # (T, B, C)

    for _ in range(200):
        optimizer.zero_grad()
        spikes = neuron(x)
        # A loss with no natural minimum in w, designed to keep pushing sigmoid(w) toward 0 or 1.
        loss = spikes.sum()
        loss.backward()
        optimizer.step()

        decay = neuron._decay()
        assert torch.all(decay > 0.0) and torch.all(decay < 1.0), f"decay left (0, 1): {decay}"

def test_bare_learnable_lif_can_leave_unit_interval() -> None:
    """
    Contrast case: plain LIFNeuron's old learnable = True path has no such guarantee: beta is a bare nn.Parameter, so nothing stops gradient
    steps from pushing it outside (0, 1). This documents why PLIFNeuron exists rather than just using LIFNeuron(learnable = True).
    """
    torch.manual_seed(0)
    neuron = LIFNeuron(beta = 0.9, learnable = True)
    optimizer = torch.optim.SGD(neuron.parameters(), lr = 1.0)

    x = torch.rand(6, 2, 4)

    left_unit_interval = False
    for _ in range(200):
        optimizer.zero_grad()
        spikes = neuron(x)
        loss = spikes.sum()
        loss.backward()
        optimizer.step()

        if not (0.0 < neuron.beta.item() < 1.0):
            left_unit_interval = True
            break

    assert left_unit_interval, "expected LIFNeuron(learnable = True)'s bare beta parameter to eventually leave (0, 1) under aggressive SGD."

def test_per_channel_decay_broadcasts_correctly() -> None:
    """
    With num_channels = 8, _decay() should have shape (8,) and broadcast cleanly against a (T, B, N, 8) input without error, producing spikes of
    the same shape.
    """
    neuron = PLIFNeuron(num_channels = 8)
    assert neuron._decay().shape == (8,)

    x = torch.rand(4, 2, 5, 8) # (T, B, N, C)
    spikes = neuron(x)
    assert spikes.shape == x.shape

def test_scalar_decay_when_num_channels_none() -> None:
    """
    With num_channels = None (the default), _decay() should be a 0-dim scalar, matching plain LIFNeuron's shared-scalar behavior.
    """
    neuron = PLIFNeuron()
    assert neuron._decay().shape == ()

def test_build_neuron_plif_threads_channels() -> None:
    """
    build_neuron("plif", channels = 16, ...) should construct a PLIFNeuron whose decay has shape (16,).
    """
    neuron = build_neuron("plif", threshold = 1.0, beta = 0.9, channels = 16)
    assert isinstance(neuron, PLIFNeuron)
    assert neuron._decay().shape == (16,)

def test_build_neuron_rejects_unknown_type() -> None:
    """
    build_neuron with an unrecognized neuron_type must raise ValueError.
    """
    try:
        build_neuron("bogus")
    except ValueError:
        return
    raise AssertionError("Expected ValueError for an unrecognized neuron_type.")
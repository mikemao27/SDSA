<div align="center">

## Spike-Driven Self-Attention

<p align="center">
  <img src="https://img.shields.io/badge/Architecture-Spiking%20Transformer-blue?style=flat-square" alt="Architecture">
  <img src="https://img.shields.io/badge/Core-Surrogate%20Gradient%20LIF-green?style=flat-square" alt="Core">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square" alt="License">
</p>

*A from-scratch, proof-of-concept spiking transformer, where the query/key/value projections and every nonlinearity are true leaky integrate-and-fire neurons instead of continuous-valued activations.*

</div>

**SDSA** (Spike-Driven Self-Attention) is a small proof-of-concept project exploring what happens when the internals of a transformer — its query/key/value projections, its attention nonlinearity, its feed-forward activations — are replaced with the discrete, event-driven spiking dynamics of biological neurons, rather than continuous-valued floating-point math. Query, key, and value are each produced by passing a linear projection's output through its own leaky integrate-and-fire (LIF) neuron, so attention in SDSA is computed directly over binary (0.0/1.0) spike trains, without a softmax. Because the spiking nonlinearity (a hard threshold) has no useful gradient almost anywhere, training relies on a hand-written surrogate-gradient `torch.autograd.Function`, which keeps the true spike in the forward pass but substitutes a smooth fast-sigmoid derivative in the backward pass.

There are a few caveats worth noting about SDSA's scope. It is validated on a single, deliberately small task — MNIST digit classification — and is not a reproduction of any specific published spiking-transformer architecture, nor a rigorous benchmark against dense (non-spiking) transformers of comparable size. The model is intentionally tiny so that it trains in minutes on a single Apple Silicon MacBook Air (CPU or the `mps` backend); it has not been evaluated on larger vision or language tasks, and it makes no attempt at low-power or neuromorphic hardware deployment. SDSA exists primarily as a hand-written learning exercise: every component (the surrogate gradient, the LIF neuron, spiking self-attention, the spiking feed-forward block) was implemented from a documented, hand-designed specification rather than ported from an existing spiking-transformer codebase.

*SDSA grew out of a broader look into spiking neural network research from accredited sources (Sandia National Laboratories, and recent brain-inspired foundation model work) over the summer. The aim here is narrow and concrete: to see, hands-on, whether a transformer's attention mechanism can be built entirely out of binary spike trains and still train end-to-end via surrogate gradients — not to advance the state of the art or make any claims about biological plausibility.*

> [!IMPORTANT]
> SDSA is restricted to the MNIST digit-classification demo included in this repository and should not be assumed to generalize to other vision or language tasks. All code is open-source with an Apache 2.0 License.

## Further Exploration
This project draws on ideas from spiking neural network research: surrogate-gradient training, leaky integrate-and-fire dynamics, and spike-based attention as explored in work from labs like Sandia National Laboratories and in recent brain-inspired foundation model efforts. SDSA does not reproduce any of that work directly; it's a small, hand-built implementation meant to make those ideas concrete and testable on a single machine.

## Contacts
Feel free to reach out with questions, corrections, or collaboration ideas. If you're interested in spiking neural networks, surrogate-gradient training, or brain-inspired approaches to transformer architectures, I'd love to hear from you.

## Citation
If you find this project useful, please give it a star and cite it via [**GitHub**](https://github.com/mikemao27/SDSA). See `LICENSE.txt` (Apache 2.0) for terms of use and attribution. We provide a sample **bibtex** citation blurb below for ease of usage. Training an attention mechanism directly over binary spike trains is far from a solved problem — surrogate gradients are an approximation, and the tradeoffs between spike-based and continuous-valued attention are still an open question. We hope this small, readable implementation is a useful starting point for others exploring the same space.

```bibtex
@software{SDSA,
  author = {Mao, Mike},
  title = {SDSA: Spike-Driven Self-Attention},
  year = {2026},
  url = {https://github.com/mikemao27/SDSA},
  version = {1.0.0}
}
```
<div align="center">

## Spike-Driven Self-Attention

<p align="center">
  <img src="https://img.shields.io/badge/Architecture-Spiking%20Transformer-blue?style=flat-square" alt="Architecture">
  <img src="https://img.shields.io/badge/Attention-Linear%20(O(N))-orange?style=flat-square" alt="Attention">
  <img src="https://img.shields.io/badge/Core-Surrogate%20Gradient%20LIF%20%2F%20PLIF-green?style=flat-square" alt="Core">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square" alt="License">
</p>

*A from-scratch spiking transformer, where the query/key/value projections and every nonlinearity are true leaky integrate-and-fire neurons instead of continuous-valued activations, with a genuinely linear, addition-only attention mechanism at its core.*

</div>

**SDSA** (Spike-Driven Self-Attention) is a research-engineering project exploring what happens when the internals of a transformer (its query/key/value projections, its attention nonlinearity, its feed-forward activations) are replaced with the discrete, event-driven spiking dynamics of biological neurons, rather than continuous-valued floating-point math. Query, key, and value are each produced by passing a linear projection's output through its own leaky integrate-and-fire (LIF or, optionally, parametric PLIF) neuron, so attention in SDSA is computed directly over binary (0.0/1.0) spike trains, without a softmax. Because there's no softmax between the two matmuls, SDSA's default attention mode reorders them into a linear, addition-only form (`Q @ (Kᵀ @ V)` rather than `(Q @ Kᵀ) @ V`), a real complexity win, not just a binary-valued version of ordinary attention, with the original quadratic form kept alongside it for direct comparison. Because the spiking nonlinearity (a hard threshold) has no useful gradient almost anywhere, training relies on a hand-written surrogate-gradient `torch.autograd.Function`, which keeps the true spike in the forward pass but substitutes a smooth fast-sigmoid derivative in the backward pass.

Beyond the core architecture, SDSA includes: a theoretical energy/synaptic-operation accounting layer that estimates the accumulate-vs-multiply-accumulate energy cost of a forward pass and its actual spike sparsity, so the usual "spiking networks are more efficient" claim is something the repo can quantify rather than merely assert; a matched-parameter dense (non-spiking) transformer baseline sharing the same tokenization pipeline, for a fair accuracy/energy comparison; support for three datasets spanning two input modalities: MNIST and CIFAR-10 (static images, rate- or direct-coded into spike trains) and Spiking Heidelberg Digits (SHD), a genuinely event-based neuromorphic audio dataset that needs no artificial spike-coding at all; and an ablation/sweep harness for comparing neuron types, coding schemes, attention modes, and timestep counts against each other.

There are a few caveats worth noting about SDSA's scope. It is not a reproduction of any specific published spiking-transformer architecture (e.g. Spikformer or the Spike-Driven Transformer line of work), nor a claim to advance the state of the art: it borrows their ideas and re-derives them from scratch as a hands-on exercise. Its energy accounting is a standard, widely-cited analytic simplification (fixed per-operation energy constants, sparsity-discounted by measured firing rate), not a cycle-accurate hardware or neuromorphic-chip simulation. Models are kept intentionally small so that every experiment (training a single configuration, or running the full ablation sweep) stays comfortably runnable on a CPU in a reasonable amount of time; none of this has been evaluated at a scale larger than that, and it makes no attempt at actual low-power or neuromorphic hardware deployment.

*SDSA grew out of a broader look into spiking neural network research from accredited sources (Sandia National Laboratories, and recent brain-inspired foundation model work). The aim here is concrete: to see, hands-on, whether a transformer's attention mechanism can be built entirely out of on/off signals, made to actually take advantage of that sparsity for a genuine efficiency argument, and still be trainable end-to-end, without making any claims about how closely this resembles real biological brains.*

> [!IMPORTANT]
> SDSA covers MNIST, CIFAR-10, and SHD (see above), each runnable on CPU, alongside a dense ANN baseline and an ablation/energy-accounting harness: but it remains a hands-on research-engineering project rather than a peer-reviewed benchmark suite, and shouldn't be assumed to generalize beyond the datasets and scales it's actually been run on. All code is open-source with an Apache 2.0 License.

## Further Exploration
This project builds on ideas from spiking neural network research: neurons that communicate in short pulses instead of continuous numbers, tricks that make those pulses trainable, and the growing interest in brain-inspired designs for large models, drawing especially on work from Sandia National Laboratories and recent brain-inspired foundation model research. SDSA doesn't reproduce any of that work directly. It's a hands-on project meant to test those ideas concretely: does a transformer actually get cheaper to run when its attention is built out of pulses instead of ordinary numbers, and how does it compare, in both accuracy and estimated energy use, to a regular transformer of the same size?

## Contacts
Feel free to reach out with questions, corrections, or collaboration ideas. If you're interested in spiking neural networks or brain-inspired approaches to transformers, I'd love to hear from you.

## Citation
If you find this project useful, please give it a star and cite it via [**GitHub**](https://github.com/mikemao27/SDSA). See `LICENSE.txt` (Apache 2.0) for terms of use and attribution. We provide a sample **bibtex** citation blurb below for ease of usage. Building attention out of pulses instead of ordinary numbers is still far from a solved problem, and how it really stacks up against a regular transformer, in accuracy and in the energy it would take to run, remains an open question. We hope this implementation, along with the experiments and energy-estimation tools built alongside it, is a useful starting point for others exploring the same space.

```bibtex
@software{SDSA,
  author = {Mao, Mike},
  title = {SDSA: Spike-Driven Self-Attention},
  year = {2026},
  url = {https://github.com/mikemao27/SDSA},
  version = {1.0.0}
}
```
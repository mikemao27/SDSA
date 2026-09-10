"""
Spiking self-attention: the spiking analogue of multi-head self-attention.

Instead of computing Q, K, V as continuous-valued tensors and combining them with a softmax, spiking self-attention projects the input into Q, K, V
currents, passes each through its own LIF neuron to obtain true binary spike tensors, and then combines those binary Q, K, V spike trains into an
attention output: without a softmax, since spikes are already non-negative and sparse.
"""

import torch
import torch.nn as nn

from .neurons import LIFNeuron

class SpikingSelfAttention(nn.Module):
    """
    Multi-head self-attention computed entirely over binary spike trains.

    The pipeline, applied identically at every one of the T time-steps in the input, is as follows. First, linearly project the input into
    query/key/value currents. Pass each of the three current tensors through its own LIF neuron to obtain binary spike tensors Q, K, V (each in
    {0.0, 1.0}). Split Q, K, V across num_heads attention heads and combine them into an attention output, scaled by attn_scale. Merge the heads,
    project back to embed_dim, and pass the result through a final output LIF neuron.

    Since there is no softmax between the two matmuls, (Q @ K^T) @ V and Q @ (K^T @ V) are exactly equal by matrix-multiply associativity (and the
    scalar attention_scale commutes through either grouping). attention_mode selects which grouping is actually computed: "quadratic" evaluates
    Q @ K^T first (an (N, N) attention matrix, O(N^2 * head_dim) per head), matching ordinary softmax-attention complexity. "linear" evaluates
    K^T @ V first (a (head_dim, head_dim) matrix, O(N * head_dim^2) per head), this is the actual spike-driven-attention trick: linear in
    sequence length N, and addition-only since Q/K/V are binary. Both modes are kept so they can be benchmarked against each other; "linear" is the
    default since it is the mode with a real efficiency claim.

    embed_dim is the size of the input/output embedding dimension. num_heads is the number of attention heads, it must evenly divide embed_dim.
    threshold is the firing threshold shared by the Q/K/V/output LIF neurons. beta is the leak factor shared by the Q/K/V/output LIF neurons.
    qkv_bias is whether the Q/K/V linear projections use a bias term. attn_scale is the scaling factor applied to attention scores before combining
    with V (analogous to the 1 / sqrt(d_k) scaling used in standard softmax attention). attention_mode is either "linear" or "quadratic" (see above).
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        threshold: float = 1.0,
        beta: float = 0.9,
        qkv_bias: bool = False,
        attention_scale: float = 0.125,
        attention_mode: str = "linear",
    ) -> None:
        if attention_mode not in {"linear", "quadratic"}:
            raise ValueError(f"attention_mode must be either 'linear' or 'quadratic'; we got {attention_mode!r} instead.")

        super().__init__()
        assert embed_dim % num_heads == 0, f"num_heads must be a factor of embed_dim; we got embed_dim: {embed_dim} and num_heads: {num_heads} instead."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.threshold = threshold
        self.beta = beta
        self.qkv_bias = qkv_bias
        self.attention_scale = attention_scale
        self.attention_mode = attention_mode

        self.head_dim = self.embed_dim // self.num_heads

        self.q_linear = nn.Linear(self.embed_dim, self.embed_dim, bias = self.qkv_bias)
        self.k_linear = nn.Linear(self.embed_dim, self.embed_dim, bias = self.qkv_bias)
        self.v_linear = nn.Linear(self.embed_dim, self.embed_dim, bias = self.qkv_bias)

        self.q_lif = LIFNeuron(threshold = self.threshold, beta = self.beta)
        self.k_lif = LIFNeuron(threshold = self.threshold, beta = self.beta)
        self.v_lif = LIFNeuron(threshold = self.threshold, beta = self.beta)

        self.proj_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_lif = LIFNeuron(threshold = self.threshold, beta = self.beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spiking self-attention over a spike-train input.

        x: is the input tensor of shape (T, B, N, embed_dim), where T is the number of time-steps and N is the number of tokens (patches) per image.
        This returns a tensor of shape (T, B, N, embed_dim): the attention output spike train, ready to be added back into the residual stream.

        We unpack T, B, N, C = x.shape (C == embed_dim). We project it Q/K/V currents (nn.Linear applies it to the last dimension regardless of the
        leading (T, B, N) dimensions), so this is fine even on a 4-D tensor. We then convert each current stream into true binary spikes by passing it
        through its LIF neuron (which internally loops over the T axis). Then, we split each current stream into heads and move the head dim next to N
        for batched matmul. Depending on attention_mode we either form the (N, N) attention matrix first ("quadratic") or the (head_dim, head_dim)
        matrix K^T @ V first ("linear"): both are exactly equal since there is no softmax in between, but the latter is O(N) instead of O(N^2) in
        the number of tokens. We merge the heads back, project to embed_dim, and spike once more, and return the output.
        """
        T, B, N, C = x.shape
        q_current = self.q_linear(x)
        k_current = self.k_linear(x)
        v_current = self.v_linear(x)

        Q = self.q_lif(q_current)
        K = self.k_lif(k_current)
        V = self.v_lif(v_current)

        Q = Q.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        K = K.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        V = V.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        if self.attention_mode == "quadratic":
            attention = (Q @ K.transpose(-2, -1)) * self.attention_scale
            output = attention @ V
        else:
            KV = K.transpose(-2, -1) @ V
            output = (Q @ KV) * self.attention_scale

        output = output.permute(0, 1, 3, 2, 4).reshape(T, B, N, C)
        output = self.proj_lif(self.proj_linear(output))

        return output
"""
Standalone wall-clock benchmark: "linear" vs "quadratic" SpikingSelfAttention as the number of tokens N grows.

Both attention_mode settings compute an exactly equal result (see tests/test_attention_equivalence.py); this script demonstrates the actual
CPU wall-clock payoff of the O(N) "linear" grouping over the O(N^2) "quadratic" grouping as N increases.

To run this, enter the following command in terminal: "python -m experiments.benchmark_attention".
"""

import time

import torch

from transformer.attention import SpikingSelfAttention

def time_attention(attention: SpikingSelfAttention, x: torch.Tensor, num_repeats: int = 5) -> float:
    """
    Return the average wall-clock seconds for one forward pass of attention on x, averaged over num_repeats runs (after one untimed warm-up run).
    """
    attention.eval()
    with torch.no_grad():
        attention(x) # warm-up

        start = time.perf_counter()
        for _ in range(num_repeats):
            attention(x)
        elapsed = time.perf_counter() - start

    return elapsed / num_repeats

def main() -> None:
    """
    For a range of token counts N, build a quadratic and a linear SpikingSelfAttention with identical weights, time a forward pass of each on
    randomly-initialized input, and print a table of wall-clock seconds per mode alongside the speedup factor.
    """
    torch.manual_seed(0)
    embed_dim, num_heads = 64, 4
    T, B = 4, 8
    token_counts = [16, 64, 256, 1024]

    print(f"{'N':>6} | {'quadratic (s)':>14} | {'linear (s)':>12} | {'speedup':>8}")
    print("-" * 50)

    for N in token_counts:
        quadratic_attn = SpikingSelfAttention(embed_dim, num_heads, attention_mode = "quadratic")
        linear_attn = SpikingSelfAttention(embed_dim, num_heads, attention_mode = "linear")
        linear_attn.load_state_dict(quadratic_attn.state_dict())

        x = torch.randn(T, B, N, embed_dim)

        quadratic_time = time_attention(quadratic_attn, x)
        linear_time = time_attention(linear_attn, x)
        speedup = quadratic_time / linear_time

        print(f"{N:>6} | {quadratic_time:>14.5f} | {linear_time:>12.5f} | {speedup:>7.2f}x")

if __name__ == "__main__":
    main()
"""
╔══════════════════════════════════════════════════════════════════════════╗
║         PROJECT NORD v4 — Interactive Chat                              ║
║                                                                        ║
║  Commands:                                                             ║
║      /stdp on|off   — Toggle online learning                           ║
║      /stats         — Show zone & MoE statistics                        ║
║      /memory        — Show memory cortex state                          ║
║      /reset         — Clear working memory                              ║
║      /expert        — Show expert routing breakdown                     ║
║      /quit          — Exit                                              ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import sys
import time
import torch
from pathlib import Path

from nord_core_v4 import NordConfig, NordModel


def load_model(model_dir: str):
    from transformers import AutoTokenizer

    model_dir = Path(model_dir)

    # Find checkpoint
    latest = model_dir / "nord_v4_latest.pt"
    if not latest.exists():
        latest = model_dir / "nord_v4_final.pt"
    if not latest.exists():
        ckpts = sorted(model_dir.glob("nord_v4_step_*.pt"))
        latest = ckpts[-1] if ckpts else None
    if latest is None:
        # Try loading v3 format
        for name in ["nord_500m_latest.pt", "nord_latest.pt"]:
            p = model_dir / name
            if p.exists():
                latest = p
                break
    if latest is None:
        print(f"  [✗] No checkpoint found in {model_dir}")
        sys.exit(1)

    print(f"  [*] Loading: {latest.name}")
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)

    # Build config from checkpoint
    saved_cfg = ckpt.get("config", {})
    cfg = NordConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16,
    )
    for k, v in saved_cfg.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if cfg.vocab_size < tokenizer.vocab_size:
        cfg.vocab_size = tokenizer.vocab_size

    # Model
    model = NordModel(cfg)
    # Filter out persistent LIF state buffers (size mismatch is OK, they reset anyway)
    state = ckpt["model_state_dict"]
    filtered = {k: v for k, v in state.items()
                if "_v_mem_state" not in k and "_i_syn_state" not in k}
    model.load_state_dict(filtered, strict=False)
    model = model.to(cfg.device)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"  [✓] Nord v4 loaded ({total/1e6:.1f}M params)")
    print(f"  [✓] {model.count_params()}")

    return model, tokenizer, cfg


@torch.no_grad()
def generate(model, tokenizer, cfg, prompt: str,
             max_tokens: int = 200, temperature: float = 0.85,
             top_p: float = 0.9, repetition_penalty: float = 1.3,
             enable_stdp: bool = False):

    input_ids = tokenizer(
        prompt, return_tensors="pt",
        max_length=cfg.max_seq_len, truncation=True,
    ).input_ids.to(cfg.device)

    model.reset_state()

    generated = input_ids.clone()
    all_stats = {}

    t_start = time.time()

    for i in range(max_tokens):
        context = generated[:, -cfg.max_seq_len:]

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(cfg.dtype == torch.float16)):
            logits, stats = model(context, enable_stdp=enable_stdp)

        next_logits = logits[:, -1, :].float()

        # Repetition penalty
        if repetition_penalty != 1.0:
            for token_id in generated[0].unique():
                next_logits[0, token_id] /= repetition_penalty

        # Temperature
        next_logits = next_logits / max(temperature, 0.01)

        # Top-p
        probs = torch.softmax(next_logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        token = sorted_idx[0, torch.multinomial(sorted_probs[0], 1)]
        generated = torch.cat([generated, token.reshape(1, 1)], dim=1)

        if token.item() == tokenizer.eos_token_id:
            break

        all_stats = stats  # keep last stats

    elapsed = time.time() - t_start
    output = tokenizer.decode(generated[0][input_ids.shape[1]:],
                              skip_special_tokens=True)
    n_tokens = generated.shape[1] - input_ids.shape[1]
    tps = n_tokens / elapsed if elapsed > 0 else 0

    rep_score = 1.0
    if n_tokens > 5:
        out_ids = generated[0][input_ids.shape[1]:].tolist()
        unique = len(set(out_ids))
        rep_score = len(out_ids) / max(unique, 1)

    return output, n_tokens, elapsed, tps, rep_score, all_stats


def print_stats(stats: dict, cfg: NordConfig):
    print(f"\n  {'─' * 50}")
    print(f"  Zone Statistics:")

    # Spike rates
    spike_rates = stats.get("spike_rates", [])
    if spike_rates:
        print(f"    Encoder:     {spike_rates[0]:.4f}")
        for i in range(min(cfg.sensory_layers, len(spike_rates)-1)):
            print(f"    Sensory[{i}]:  {spike_rates[i+1]:.4f}")
        offset = cfg.sensory_layers + 1
        for i in range(cfg.association_layers):
            if offset + i < len(spike_rates):
                print(f"    Assoc[{i}]:    {spike_rates[offset+i]:.4f} (MoE)")
        offset += cfg.association_layers
        for i in range(cfg.executive_layers):
            if offset + i < len(spike_rates):
                print(f"    Exec[{i}]:     {spike_rates[offset+i]:.4f}")

    # MoE
    entropy = stats.get("moe_route_entropy", None)
    if entropy is not None:
        print(f"\n  MoE Routing:")
        print(f"    Entropy: {entropy:.3f}")
        for e in range(cfg.n_experts):
            load = stats.get(f"expert_{e}_load", 0)
            bar = "█" * int(load * 40)
            print(f"    Expert {e}: {load:.2%} {bar}")

    # Memory
    mem_rate = stats.get("memory_spike_rate", None)
    if mem_rate is not None:
        print(f"\n  Memory Cortex:")
        print(f"    Spike rate:  {mem_rate:.4f}")
        print(f"    Gate:        {stats.get('gate_activity', 0):.4f}")
        print(f"    Mix weight:  {stats.get('memory_mix', 0):.4f}")

    sparsity = stats.get("sparsity", 0)
    print(f"\n  Overall Sparsity: {sparsity:.1%}")
    print(f"  {'─' * 50}")


def main():
    print("═" * 60)
    print("  ⚡ PROJECT NORD v4 — Brain-Inspired SNN Chat")
    print("═" * 60)

    default_dir = "nord_v4_model"
    print(f"\n  Model directory?")
    print(f"  (Enter = {default_dir})")
    model_input = input("  Path: ").strip()
    model_dir = model_input if model_input else default_dir

    model, tokenizer, cfg = load_model(model_dir)

    stdp_enabled = False
    last_stats = {}

    print(f"\n  Commands: /stdp on|off, /stats, /memory, /expert, /reset, /quit")
    print(f"  {'─' * 50}\n")

    while True:
        try:
            user = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user:
            continue

        # Commands
        if user.lower() == "/quit":
            break
        elif user.lower() == "/stdp on":
            stdp_enabled = True
            print("  [⚙] STDP enabled")
            continue
        elif user.lower() == "/stdp off":
            stdp_enabled = False
            print("  [⚙] STDP disabled")
            continue
        elif user.lower() == "/stats":
            print_stats(last_stats, cfg)
            continue
        elif user.lower() == "/memory":
            mem_rate = last_stats.get("memory_spike_rate", "N/A")
            gate = last_stats.get("gate_activity", "N/A")
            mix = last_stats.get("memory_mix", "N/A")
            print(f"  Memory: rate={mem_rate}, gate={gate}, mix={mix}")
            continue
        elif user.lower() == "/expert":
            for e in range(cfg.n_experts):
                load = last_stats.get(f"expert_{e}_load", 0)
                bar = "█" * int(load * 40)
                print(f"    Expert {e}: {load:.2%} {bar}")
            continue
        elif user.lower() == "/reset":
            model.reset_state()
            print("  [⚙] Working memory cleared")
            continue

        # Generate
        output, n_tok, elapsed, tps, rep, stats = generate(
            model, tokenizer, cfg, user,
            enable_stdp=stdp_enabled,
        )
        last_stats = stats

        print(f"  Nord: {output}")
        sparsity = stats.get("sparsity", 0)
        print(f"  [{n_tok} tok, {elapsed:.1f}s, {tps:.1f} tok/s "
              f"[REP {rep:.1f}] [SPR {sparsity:.0%}]]")


if __name__ == "__main__":
    main()
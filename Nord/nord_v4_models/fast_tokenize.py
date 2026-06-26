"""
Fast LMDB builder — batch tokenization with visible progress
"""

import json, struct, time, lmdb
from transformers import AutoTokenizer
import numpy as np

# ── Config ──
SRC = "/nord_dataset/train_data.jsonl"
DST = "/nord_dataset/train_data_lmdb"
SEQ = 512
BATCH = 1024

# ── Init tokenizer ──
print("Loading tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
PAD_ID = tok.pad_token_id

# ── Read all texts ──
print(f"[1/3] Reading JSONL into memory...", flush=True)
t0 = time.time()
texts = []
with open(SRC, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i % 1_000_000 == 0 and i > 0:
            print(f"    read {i:,} lines...", flush=True)
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except:
            continue
        text = obj.get("text") or obj.get("content") or obj.get("passage", "")
        if len(text) >= 30:
            texts.append(text)

print(f"    {len(texts):,} valid texts in {time.time()-t0:.0f}s", flush=True)

# ── Batch tokenize ──
print(f"[2/3] Batch tokenizing {len(texts):,} texts (batch={BATCH})...", flush=True)
t1 = time.time()

env = lmdb.open(DST, map_size=80 * (1024**3))
txn = env.begin(write=True)
count = 0
total_tok = 0
total_batches = (len(texts) + BATCH - 1) // BATCH

for batch_idx in range(0, len(texts), BATCH):
    batch = texts[batch_idx : batch_idx + BATCH]
    batch_num = batch_idx // BATCH + 1

    enc = tok(
        batch,
        max_length=SEQ,
        truncation=True,
        padding="max_length",
        return_tensors="np",
        return_attention_mask=False,
    )
    ids_np = enc.input_ids.astype(np.int32)

    for j in range(ids_np.shape[0]):
        row = ids_np[j]
        non_pad = int(np.sum(row != PAD_ID))
        if non_pad < 10:
            continue
        txn.put(f"sample_{count:010d}".encode(), row.tobytes())
        count += 1
        total_tok += non_pad

    # Progress every 100 batches (~100k docs)
    if batch_num % 100 == 0 or batch_num == total_batches:
        elapsed = time.time() - t1
        pct = batch_num / total_batches * 100
        eta = (elapsed / batch_num) * (total_batches - batch_num)
        print(
            f"    [{pct:5.1f}%] batch {batch_num}/{total_batches} | "
            f"{count:,} samples | {total_tok/1e6:.0f}M tok | "
            f"{elapsed:.0f}s elapsed | ETA {eta:.0f}s",
            flush=True,
        )

    # Commit every 500k
    if count % 500_000 < BATCH and count >= 500_000:
        txn.commit()
        txn = env.begin(write=True)

# Save metadata
txn.put(b"__len__", struct.pack("<Q", count))
txn.put(b"__total_tokens__", struct.pack("<Q", total_tok))
txn.commit()
env.close()

elapsed = time.time() - t1
print(f"\n[3/3] Done!", flush=True)
print(f"    Samples:  {count:,}", flush=True)
print(f"    Tokens:   {total_tok:,} ({total_tok/1e6:.0f}M)", flush=True)
print(f"    Time:     {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
print(f"    Speed:    {count/elapsed:.0f} doc/s", flush=True)
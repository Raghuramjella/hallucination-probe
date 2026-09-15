"""STAGE 2 -- capture internal activations for each graded answer.

Two-pass design: generation already happened in stage 1. Here we re-run a
single forward pass over prompt+answer and read the hidden states. This is
far easier to reason about than extracting during generation (whose
hidden_states is a nested tuple over both steps and layers).

Output: data/acts_XXXXX.npy shards, each (n, n_layers+1, hidden_dim) float16
        plus labels.npy and groups.npy

Resumable: existing shards are skipped.
"""
import json
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from config import (ACTS_PREFIX, ANSWERS_FILE, GROUPS_FILE, LABELS_FILE,
                    POOLING, SHARD_SIZE)
from model_utils import load_model


@torch.no_grad()
def activations_for(tok, model, prompt, answer):
    """Return (n_layers+1, hidden_dim) -- one pooled vector per layer."""
    n_prompt = tok(prompt, return_tensors="pt").input_ids.shape[1]
    enc = tok(prompt + answer, return_tensors="pt").to(model.device)
    n_total = enc.input_ids.shape[1]

    if n_total <= n_prompt:          # model emitted nothing usable
        return None

    out = model(**enc, output_hidden_states=True)

    vecs = []
    for layer_h in out.hidden_states:            # (1, seq, dim) per layer
        span = layer_h[0, n_prompt:n_total, :]
        vec = span.mean(dim=0) if POOLING == "mean" else span[-1]
        vecs.append(vec.float().cpu().numpy())
    return np.stack(vecs)


def main():
    records = [json.loads(l) for l in open(ANSWERS_FILE, encoding="utf-8")]
    print(f"{len(records)} graded answers to extract from")

    tok, model = load_model()

    labels, groups = [], []
    for shard_start in range(0, len(records), SHARD_SIZE):
        shard_id = shard_start // SHARD_SIZE
        path = ACTS_PREFIX.parent / f"{ACTS_PREFIX.name}_{shard_id:05d}.npy"
        chunk = records[shard_start:shard_start + SHARD_SIZE]

        if path.exists():
            print(f"shard {shard_id} exists, skipping")
            labels += [r["label"] for r in chunk]
            groups += [r["group"] for r in chunk]
            continue

        feats, keep_labels, keep_groups = [], [], []
        for r in tqdm(chunk, desc=f"shard {shard_id}"):
            a = activations_for(tok, model, r["prompt"], r["answer"])
            if a is None:
                continue
            feats.append(a)
            keep_labels.append(r["label"])
            keep_groups.append(r["group"])

        np.save(path, np.stack(feats).astype(np.float16))
        labels += keep_labels
        groups += keep_groups
        print(f"saved {path.name}  shape={np.stack(feats).shape}")

    np.save(LABELS_FILE, np.array(labels, dtype=np.int64))
    np.save(GROUPS_FILE, np.array(groups, dtype=object), allow_pickle=True)
    print(f"\ndone. labels={len(labels)}  positives(wrong)={int(sum(labels))}")
    print("From here on you never need the GPU again.")


if __name__ == "__main__":
    main()

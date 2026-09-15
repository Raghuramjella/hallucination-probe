"""THE CRITICAL COMPARISON -- does the probe beat the free alternative?

The first question anyone will ask is "couldn't you just use the model's own
probability?". If you cannot answer that with a number, the project does not
stand up.

This computes token log-probabilities for each already-generated answer (one
GPU forward pass over prompt+answer -- no regeneration) and scores them as
hallucination detectors, so they can be put head-to-head with the probe.

    python src/baselines.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import ANSWERS_FILE, ARTIFACTS, DATA
from model_utils import load_model

LOGPROB_FILE = DATA / "logprobs.npz"


@torch.no_grad()
def answer_logprobs(tok, model, prompt, answer):
    """Per-token log P for the answer tokens only."""
    n_prompt = tok(prompt, return_tensors="pt").input_ids.shape[1]
    enc = tok(prompt + answer, return_tensors="pt").to(model.device)
    ids = enc.input_ids
    if ids.shape[1] <= n_prompt:
        return None

    logits = model(**enc).logits[0]                 # (seq, vocab)
    logprobs = torch.log_softmax(logits.float(), dim=-1)

    # token at position t is predicted by logits at position t-1
    tgt = ids[0, n_prompt:]
    src = logprobs[n_prompt - 1:-1]
    return src.gather(1, tgt.unsqueeze(1)).squeeze(1).cpu().numpy()


def compute():
    records = [json.loads(l) for l in open(ANSWERS_FILE, encoding="utf-8")]
    tok, model = load_model()

    mean_lp, min_lp, entropy_proxy, labels = [], [], [], []
    for r in tqdm(records, desc="log-probs"):
        lp = answer_logprobs(tok, model, r["prompt"], r["answer"])
        if lp is None or len(lp) == 0:
            continue
        mean_lp.append(float(lp.mean()))
        min_lp.append(float(lp.min()))
        entropy_proxy.append(float(-lp.mean()))     # higher = less certain
        labels.append(r["label"])

    np.savez(LOGPROB_FILE, mean_lp=np.array(mean_lp), min_lp=np.array(min_lp),
             entropy=np.array(entropy_proxy), labels=np.array(labels))
    print("saved ->", LOGPROB_FILE)


def evaluate():
    d = np.load(LOGPROB_FILE)
    y = d["labels"]

    # a hallucination score: HIGHER should mean MORE likely wrong,
    # so log-probability is negated.
    rows = [
        ("mean token log-prob", -d["mean_lp"]),
        ("min token log-prob", -d["min_lp"]),
        ("mean negative log-prob", d["entropy"]),
    ]
    print("\n%-26s %8s %14s" % ("baseline", "AUROC", "extra fwd passes"))
    print("-" * 52)
    for name, score in rows:
        print("%-26s %8.4f %14d" % (name, roc_auc_score(y, score), 0))

    probe_path = ARTIFACTS / "probe.joblib"
    if probe_path.exists():
        import joblib
        saved = joblib.load(probe_path)
        print("%-26s %8.4f %14d" % ("internal-state probe",
                                    saved["auroc"], 0))
        print("\nThe probe costs one dot product on activations already "
              "computed.\nSelf-consistency methods cost 10 extra generations "
              "for comparable quality.\nThat ratio is the headline.")


if __name__ == "__main__":
    if not LOGPROB_FILE.exists():
        compute()
    evaluate()

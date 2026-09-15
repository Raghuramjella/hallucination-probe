"""Smoke test: run every non-GPU code path on synthetic data.

Catches runtime bugs in grading and in the probe pipeline before you spend
GPU time discovering them. Run this after any edit to grading.py or probe.py.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

fails = []


def check(name, got, want):
    ok = got == want
    print("  %-52s %s" % (name, "ok" if ok else "FAIL got=%r want=%r" % (got, want)))
    if not ok:
        fails.append(name)


# ---------------------------------------------------------------- grading ---
print("\n[1] grading")
from grading import group_key, is_correct, normalize

check("normalize strips articles/punctuation",
      normalize("The  Eiffel Tower!"), "eiffel tower")
check("exact answer graded correct",
      is_correct("Paris", ["Paris", "Paris, France"]), True)
check("answer inside a sentence graded correct",
      is_correct("The capital is Paris.", ["Paris"]), True)
check("wrong answer graded wrong",
      is_correct("London", ["Paris"]), False)
check("empty answer graded wrong",
      is_correct("", ["Paris"]), False)
check("2-char alias ignored (too short to trust)",
      is_correct("a totally unrelated answer", ["UK"]), False)
check("group_key normalises the entity",
      group_key(["The Eiffel Tower"]), "eiffel tower")
check("group_key survives empty aliases",
      group_key([]), "__unknown__")


# ------------------------------------------------------------------ probe ---
print("\n[2] probe pipeline on synthetic activations")
from probe import fit_probe, grouped_split
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
N, LAYERS, DIM = 400, 6, 32

y = rng.integers(0, 2, N)
groups = np.array(["entity_%d" % (i % 50) for i in range(N)])

# plant a real signal in layer 3 only, noise everywhere else
X = rng.normal(size=(N, LAYERS, DIM)).astype(np.float16)
X[:, 3, :] += (y[:, None] * 1.5).astype(np.float16)

tr, te = grouped_split(y, groups)
overlap = set(groups[tr]) & set(groups[te])
check("train/test entity groups are disjoint", len(overlap), 0)
check("split covers every example", len(tr) + len(te), N)

aucs = []
for L in range(LAYERS):
    Xl = X[:, L, :].astype(np.float32)
    clf = fit_probe(Xl[tr], y[tr])
    aucs.append(roc_auc_score(y[te], clf.predict_proba(Xl[te])[:, 1]))

best = int(np.argmax(aucs))
print("     per-layer AUROC:", [round(a, 3) for a in aucs])
check("sweep finds the planted signal at layer 3", best, 3)
check("planted layer scores high", bool(aucs[3] > 0.85), True)
check("noise layers score near chance", bool(max(aucs[:3] + aucs[4:]) < 0.70), True)


# -------------------------------------------------------------- selective ---
print("\n[3] risk-coverage curve")
from selective import risk_coverage

wrong = np.array([0, 0, 0, 1, 1])
risk = np.array([0.1, 0.2, 0.3, 0.8, 0.9])   # perfect ordering
cov, acc = risk_coverage(risk, wrong)
check("full coverage accuracy = base rate", round(float(acc[-1]), 3), 0.6)
check("perfect ranking gives 100% at 60% coverage", round(float(acc[2]), 3), 1.0)
check("coverage ends at 1.0", round(float(cov[-1]), 3), 1.0)


# ------------------------------------------------------------------ config ---
print("\n[4] config + imports")
import config
check("model id set", isinstance(config.MODEL_ID, str) and len(config.MODEL_ID) > 0, True)
check("data dir created", config.DATA.exists(), True)
try:
    import extract, generate, model_utils  # noqa: F401
    check("generate/extract/model_utils import cleanly", True, True)
except ModuleNotFoundError as e:
    # transformers/torch are GPU-side deps -- absent locally is expected
    print("  %-52s skipped (%s not installed here)" % (
        "GPU-side modules", e.name))


print("\n" + "=" * 62)
if fails:
    print("FAILED: " + ", ".join(fails))
    sys.exit(1)
print("ALL CHECKS PASSED - the non-GPU logic is sound.")
print("Remaining risk is GPU-side only: model loading and generation.")

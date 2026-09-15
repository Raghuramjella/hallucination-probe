"""STAGES 3 & 4 -- train the probe and sweep every layer.

Runs on CPU in seconds. This is where all the iteration happens.

Output: figures/layer_sweep.png, artifacts/probe.joblib, printed AUROC table.
"""
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from config import (ACTS_PREFIX, ARTIFACTS, FIGURES, GROUPS_FILE, LABELS_FILE,
                    RANDOM_SEED, TEST_FRACTION)


def load_all():
    shards = sorted(ACTS_PREFIX.parent.glob(f"{ACTS_PREFIX.name}_*.npy"))
    if not shards:
        raise SystemExit("no activation shards found -- run extract.py first")
    X = np.concatenate([np.load(s) for s in shards], axis=0)
    y = np.load(LABELS_FILE)
    g = np.load(GROUPS_FILE, allow_pickle=True)
    n = min(len(X), len(y), len(g))
    return X[:n], y[:n], g[:n]


def grouped_split(y, groups):
    """Split by ENTITY, not at random.

    A random split puts the same entity in train and test. The probe then
    learns 'questions about Paris are usually answered right' instead of
    learning truthfulness, and AUROC inflates. This is the single most
    common way probing results turn out to be worthless.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION,
                            random_state=RANDOM_SEED)
    return next(gss.split(np.zeros(len(y)), y, groups))


def fit_probe(Xtr, ytr):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=1.0),
    ).fit(Xtr, ytr)


def main():
    X, y, groups = load_all()
    n_layers = X.shape[1]
    print(f"X={X.shape}  hallucination rate={y.mean():.3f}  "
          f"unique entities={len(set(groups))}")

    tr, te = grouped_split(y, groups)
    print(f"train={len(tr)}  test={len(te)}  (entity-disjoint)")

    scores = []
    for L in range(n_layers):
        Xl = X[:, L, :].astype(np.float32)
        clf = fit_probe(Xl[tr], y[tr])
        auc = roc_auc_score(y[te], clf.predict_proba(Xl[te])[:, 1])
        scores.append(auc)
        print(f"  layer {L:3d}  AUROC {auc:.4f}")

    best = int(np.argmax(scores))
    print(f"\nBEST: layer {best}  AUROC {scores[best]:.4f}  "
          f"({100 * best / (n_layers - 1):.0f}% depth)")
    interpret(scores[best])

    # refit and persist the winner
    Xb = X[:, best, :].astype(np.float32)
    clf = fit_probe(Xb[tr], y[tr])
    joblib.dump({"model": clf, "layer": best, "auroc": scores[best]},
                ARTIFACTS / "probe.joblib")

    # leakage check: random split should score HIGHER -- that gap is the leak
    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(len(y))
    cut = int(len(y) * (1 - TEST_FRACTION))
    rtr, rte = perm[:cut], perm[cut:]
    rclf = fit_probe(Xb[rtr], y[rtr])
    rauc = roc_auc_score(y[rte], rclf.predict_proba(Xb[rte])[:, 1])
    print(f"\nrandom-split AUROC {rauc:.4f} vs entity-split {scores[best]:.4f}")
    print(f"leakage inflation = {rauc - scores[best]:+.4f}  "
          "(this gap is exactly what a careless split would have hidden)")

    plot(scores, best)


def interpret(auc):
    if auc < 0.55:
        msg = "~chance. Check labels, try a 3B model, or the signal isn't there."
    elif auc < 0.65:
        msg = "weak but real. Worth pushing on."
    elif auc < 0.75:
        msg = "GOOD -- real usable signal. You're in business."
    elif auc < 0.90:
        msg = "STRONG. This is a genuinely useful detector."
    else:
        msg = "SUSPICIOUS -- assume a leak until you have proved otherwise."
    print(f"  verdict: {msg}")


def plot(scores, best):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(scores, marker="o", ms=3, lw=1.6, color="#1b3a5c")
    ax.axhline(0.5, ls="--", lw=1, color="#999", label="chance")
    ax.axvline(best, ls=":", lw=1.2, color="#a4551b",
               label=f"best = layer {best}")
    ax.set_xlabel("layer"); ax.set_ylabel("AUROC")
    ax.set_title("Where does the truthfulness signal live?")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    out = FIGURES / "layer_sweep.png"
    fig.savefig(out, dpi=160)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()

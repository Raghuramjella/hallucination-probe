"""STAGE 5 -- turn the probe into the number that sells the project.

'If the system declines the riskiest 20% of questions, accuracy on the rest
 rises from X% to Y%.'  That sentence is the whole pitch.
"""
import sys

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from config import ARTIFACTS, FIGURES
from probe import fit_probe, grouped_split, load_all


def risk_coverage(risk, wrong):
    """Sort by risk, answer the safest first, track accuracy as we go."""
    order = np.argsort(risk)
    correct = 1 - wrong[order]
    cov = np.arange(1, len(order) + 1) / len(order)
    acc = np.cumsum(correct) / np.arange(1, len(order) + 1)
    return cov, acc


def main():
    X, y, groups = load_all()
    saved = joblib.load(ARTIFACTS / "probe.joblib")
    L = saved["layer"]
    tr, te = grouped_split(y, groups)

    Xl = X[:, L, :].astype(np.float32)
    clf = fit_probe(Xl[tr], y[tr])
    risk = clf.predict_proba(Xl[te])[:, 1]
    wrong = y[te]

    print(f"layer {L}   AUROC {roc_auc_score(wrong, risk):.4f}")
    base = 1 - wrong.mean()
    print(f"answer-everything accuracy: {100 * base:.1f}%\n")

    cov, acc = risk_coverage(risk, wrong)
    aurc = float(np.trapz(1 - acc, cov))
    print(f"{'coverage':>10} {'accuracy':>10} {'gain':>8}")
    for target in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        i = max(0, int(target * len(cov)) - 1)
        print(f"{100 * cov[i]:9.0f}% {100 * acc[i]:9.1f}% "
              f"{100 * (acc[i] - base):+7.1f}")
    print(f"\nAURC (lower is better): {aurc:.4f}")

    i80 = max(0, int(0.8 * len(cov)) - 1)
    print("\n>>> HEADLINE: declining the riskiest 20% raises accuracy "
          f"from {100 * base:.1f}% to {100 * acc[i80]:.1f}%")

    # cheap baseline for comparison: does the probe beat random ordering?
    rng = np.random.default_rng(0)
    _, racc = risk_coverage(rng.random(len(wrong)), wrong)
    print(f"    (random ordering at 80% coverage: {100 * racc[i80]:.1f}%)")

    plot(cov, acc, base)


def plot(cov, acc, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(100 * cov, 100 * acc, lw=2, color="#1e6b4a")
    ax.axhline(100 * base, ls="--", lw=1, color="#a4551b",
               label=f"answer everything ({100*base:.1f}%)")
    ax.set_xlabel("coverage: % of questions answered")
    ax.set_ylabel("accuracy on answered questions (%)")
    ax.set_title("Abstaining on risky questions raises accuracy")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    out = FIGURES / "risk_coverage.png"
    fig.savefig(out, dpi=160)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()

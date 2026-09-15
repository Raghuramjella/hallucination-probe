"""Error bars and the honest leakage test.

A single split gives one number with no uncertainty. Repeating over seeds
turns "0.8116" into "0.81 +/- 0.02", which is the difference between a claim
and an anecdote -- and it settles whether the random-vs-entity split gap is
real or noise.

CPU only, about a minute.

    python src/robustness.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
from config import TEST_FRACTION
from probe import fit_probe, load_all

N_SEEDS = 10


def entity_split(y, groups, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=seed)
    return next(gss.split(np.zeros(len(y)), y, groups))


def random_split(y, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    cut = int(len(y) * (1 - TEST_FRACTION))
    return perm[:cut], perm[cut:]


def score(X, y, tr, te):
    clf = fit_probe(X[tr], y[tr])
    return roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])


def main():
    X, y, groups = load_all()

    # pick the best layer once, on seed 0, then measure it across seeds
    tr0, te0 = entity_split(y, groups, 0)
    aucs0 = [score(X[:, L, :].astype(np.float32), y, tr0, te0)
             for L in range(X.shape[1])]
    best = int(np.argmax(aucs0))
    Xb = X[:, best, :].astype(np.float32)
    print("evaluating layer %d over %d seeds\n" % (best, N_SEEDS))

    ent, rnd = [], []
    for s in range(N_SEEDS):
        ent.append(score(Xb, y, *entity_split(y, groups, s)))
        rnd.append(score(Xb, y, *random_split(y, s)))
        print("  seed %2d   entity %.4f   random %.4f" % (s, ent[-1], rnd[-1]))

    ent, rnd = np.array(ent), np.array(rnd)
    print("\n%-16s %.4f +/- %.4f" % ("entity split", ent.mean(), ent.std()))
    print("%-16s %.4f +/- %.4f" % ("random split", rnd.mean(), rnd.std()))

    diff = rnd.mean() - ent.mean()
    pooled = np.sqrt(ent.std() ** 2 + rnd.std() ** 2)
    print("\ndifference (random - entity): %+.4f" % diff)
    if abs(diff) < pooled:
        print("-> within noise. Conclusion: NO measurable entity leakage.")
        print("   Do not claim grouped splits 'help' -- claim leakage is absent.")
    else:
        print("-> larger than the spread. Real effect; report it as such.")

    # how much of the oracle's achievable gain does the probe capture?
    base = 1 - y.mean()
    print("\ncoverage   probe    oracle   captured")
    tr, te = entity_split(y, groups, 0)
    clf = fit_probe(Xb[tr], y[tr])
    risk = clf.predict_proba(Xb[te])[:, 1]
    wrong = y[te]
    order = np.argsort(risk)
    acc = np.cumsum(1 - wrong[order]) / np.arange(1, len(order) + 1)
    b = 1 - wrong.mean()
    for c in (0.9, 0.8, 0.7, 0.6, 0.5):
        i = max(0, int(c * len(acc)) - 1)
        oracle = min(1.0, b / c)
        got = acc[i]
        cap = (got - b) / (oracle - b) if oracle > b else float("nan")
        print("   %3.0f%%   %6.1f%%  %6.1f%%    %5.0f%%"
              % (100 * c, 100 * got, 100 * oracle, 100 * cap))
    print("\n(base accuracy %.1f%%)" % (100 * base))


if __name__ == "__main__":
    main()

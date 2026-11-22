"""CROSS-MODEL PROBE TRANSFER (RQ2) -- does a truthfulness probe trained on
model A detect hallucinations of model B?

Different models live in different activation spaces (different width,
arbitrary basis), so a probe vector cannot be carried across directly. We fit
a linear map B -> A on paired stimuli (the same questions shown to both
models), push model B's activations through it, and score them with model A's
probe -- against model B's OWN correctness labels.

Reads the cache written by notebooks/02_improvements.ipynb:
    <root>/<model-name>/trivia_feats_*.npz   (a_mean, a_last, p_last, label, group, min_lp, idx)
    <root>/<model-name>/results.json         (best_layer, n_layers)
Questions are paired across models by `idx` (position in the TriviaQA slice,
identical for every model run with the same N_TRIVIA).

Splits (entity-disjoint, re-drawn per seed):
    P  probe-training questions      (labels of A used here, and of B for the upper bound)
    M  map-fitting questions         (NO labels used -- activations only)
    E  evaluation questions          (B's labels used only for scoring)

Methods scored on E against B's labels:
    native        probe trained on B itself                     -> upper bound
    ridge         A's probe on ridge-mapped B activations       -> transfer
    procrustes    A's probe on PCA + orthogonal-Procrustes map  -> transfer (rotation only)
Controls -- transfer means nothing unless it clearly beats these:
    shuffled-pair ridge     map fitted on mismatched question pairs
    random orthogonal       Procrustes with random rotations (50 per split -> null
                            distribution + permutation p-value)
    unaligned               A's probe on raw B activations (truncated/zero-padded)
Difficulty baselines -- does transfer carry more than "this question is hard"?
    A probe on A            A's probe on A's OWN activations, scored vs B's labels
    A correctness           whether A got the question right, scored vs B's labels

CPU only. Usage:
    python src/transfer.py --root /content/drive/MyDrive/hallucination_probe_v2
    python src/transfer.py --root ... --models Qwen2.5-1.5B-Instruct Qwen2.5-3B-Instruct --sweep
"""
import argparse
import json
from itertools import permutations
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

POOLINGS = ("a_mean", "a_last", "p_last")


# ------------------------------------------------------------------ data ---
def model_dirs(root, names=None):
    dirs = sorted(p.parent for p in Path(root).glob("*/trivia_feats_00000.npz"))
    if names:
        by_name = {d.name: d for d in dirs}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise SystemExit(f"no feature cache for {missing} under {root}; found {list(by_name)}")
        dirs = [by_name[n] for n in names]
    return dirs


def model_info(d, pooling=None):
    """best_layer is the validation-chosen layer FOR THIS POOLING (notebook 02 stores one per
    pooling in results["layers"]); results["best_layer"] belongs to the best pooling only."""
    f = d / "results.json"
    r = json.loads(f.read_text()) if f.exists() else {}
    layer = (r.get("layers") or {}).get(pooling) if pooling else None
    return {"name": d.name, "best_layer": layer if layer is not None else r.get("best_layer"),
            "n_layers": r.get("n_layers"), "best_pooling": r.get("best_pooling")}


def load_feats(d, pooling, layers):
    """layers: int -> X is (n, d); list -> X is (n, len(layers), d). float32."""
    X, meta = [], {k: [] for k in ("label", "group", "min_lp", "idx")}
    for shard in sorted(d.glob("trivia_feats_*.npz")):
        z = np.load(shard)
        if len(z["idx"]) == 0:
            continue
        X.append(z[pooling][:, layers].astype(np.float32))
        for k in meta:
            meta[k].append(z[k])
    out = {k: np.concatenate(v) for k, v in meta.items()}
    out["X"] = np.concatenate(X)
    return out


def pair(a, b):
    """Restrict two models' data to the questions both answered, same order."""
    common = np.intersect1d(a["idx"], b["idx"])
    ra = {q: i for i, q in enumerate(a["idx"])}
    rb = {q: i for i, q in enumerate(b["idx"])}
    ia = np.array([ra[q] for q in common])
    ib = np.array([rb[q] for q in common])
    return ({k: v[ia] for k, v in a.items()}, {k: v[ib] for k, v in b.items()})


def three_way_split(groups, seed, p_frac=0.4, m_frac=0.3):
    """Entity-disjoint P / M / E index sets."""
    n = len(groups)
    rest, E = next(GroupShuffleSplit(1, test_size=1 - p_frac - m_frac, random_state=seed)
                   .split(np.zeros(n), groups=groups))
    ip, im = next(GroupShuffleSplit(1, test_size=m_frac / (p_frac + m_frac), random_state=seed + 1000)
                  .split(np.zeros(len(rest)), groups=groups[rest]))
    return rest[ip], rest[im], E


# ---------------------------------------------------------------- models ---
def fit_probe(X, y):
    # identical to notebook 02, so "native" numbers match the per-model results
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(X, y)


def risk(clf, X):
    return clf.predict_proba(X)[:, 1]


class RidgeMap:
    """Unconstrained linear map B -> A, regularisation chosen by efficient LOO-CV."""

    def fit(self, Xb, Xa):
        self.sb = StandardScaler().fit(Xb)
        self.r = RidgeCV(alphas=np.logspace(0, 6, 13)).fit(self.sb.transform(Xb), Xa)
        return self

    def transform(self, Xb):
        return self.r.predict(self.sb.transform(Xb))


class ProcrustesMap:
    """PCA both spaces to r dims, then scaled orthogonal rotation B -> A.

    random=True replaces the fitted rotation with a random orthogonal matrix:
    the single most important control -- same geometry, no correspondence.
    """

    def __init__(self, r=256, random=False, seed=0):
        self.r, self.random, self.seed = r, random, seed

    def fit(self, Xb, Xa):
        r = min(self.r, Xa.shape[1], Xb.shape[1], len(Xa) - 1)
        self.pa = PCA(r, random_state=0).fit(Xa)
        self.pb = PCA(r, random_state=0).fit(Xb)
        A, B = self.pa.transform(Xa), self.pb.transform(Xb)
        self.r_, self.s_rand = r, np.linalg.norm(A) / np.linalg.norm(B)
        if self.random:
            q, _ = np.linalg.qr(np.random.default_rng(self.seed).normal(size=(r, r)))
            self.R, self.s = q, np.linalg.norm(A) / np.linalg.norm(B)
        else:
            U, S, Vt = np.linalg.svd(B.T @ A)
            self.R, self.s = U @ Vt, S.sum() / (B ** 2).sum()
        return self

    def transform(self, Xb):
        return self.pa.inverse_transform(self.s * self.pb.transform(Xb) @ self.R)

    def random_transforms(self, Xb, n, seed):
        """Yield Xb mapped by n random rotations -- the null distribution."""
        Zb = self.pb.transform(Xb)
        rng = np.random.default_rng(seed)
        for _ in range(n):
            q, _ = np.linalg.qr(rng.normal(size=(self.r_, self.r_)))
            yield self.pa.inverse_transform(self.s_rand * Zb @ q)


def unaligned(Xb, d_a):
    if Xb.shape[1] >= d_a:
        return Xb[:, :d_a]
    return np.hstack([Xb, np.zeros((len(Xb), d_a - Xb.shape[1]), Xb.dtype)])


def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


# ------------------------------------------------------------- transfer ---
N_NULL = 50     # random rotations per split for the null distribution

METHODS = ["native B probe (upper bound)", "transfer: ridge map", "transfer: Procrustes map",
           "control: shuffled-pair ridge", "control: random orthogonal (mean of null)", "control: unaligned",
           "difficulty: A probe on A", "difficulty: A correctness", "B min token log-prob"]


def transfer_one_seed(A, B, seed, pca_r=256):
    """A, B: paired dicts with X (n, d) at one layer each. Returns {method: AUROC} on E."""
    P, M, E = three_way_split(A["group"], seed)
    XA, XB, yA, yB = A["X"], B["X"], A["label"], B["label"]

    gA = fit_probe(XA[P], yA[P])
    gB = fit_probe(XB[P], yB[P])

    ridge = RidgeMap().fit(XB[M], XA[M])
    proc = ProcrustesMap(pca_r).fit(XB[M], XA[M])
    perm = np.random.default_rng(seed).permutation(len(M))
    ridge_shuf = RidgeMap().fit(XB[M], XA[M][perm])

    y = yB[E]
    null = [safe_auc(y, risk(gA, Xm)) for Xm in proc.random_transforms(XB[E], N_NULL, seed)]
    scores = {
        METHODS[0]: risk(gB, XB[E]),
        METHODS[1]: risk(gA, ridge.transform(XB[E])),
        METHODS[2]: risk(gA, proc.transform(XB[E])),
        METHODS[3]: risk(gA, ridge_shuf.transform(XB[E])),
        METHODS[4]: None,
        METHODS[5]: risk(gA, unaligned(XB[E], XA.shape[1])),
        METHODS[6]: risk(gA, XA[E]),
        METHODS[7]: yA[E].astype(float),
        METHODS[8]: -B["min_lp"][E],
    }
    aucs = {k: safe_auc(y, v) for k, v in scores.items() if v is not None}
    aucs[METHODS[4]] = float(np.mean(null))
    aucs["_null"] = null
    return aucs, scores, y


def verdict(summary, null):
    """Decision rule fixed in advance; uses only tests that hold across ALL seeds.

    (The single 'unaligned' draw and the seed-0 bootstraps are descriptive only:
    one arbitrary basis or one split can land far from 0.5 by chance.)
    """
    best_k = max(METHODS[1:3], key=lambda k: summary[k]["mean"])
    t, sd = summary[best_k]["mean"], summary[best_k]["std"]
    checks = {
        "beats random-rotation null on every split (perm. p < 0.05)": null["perm_p_value_max_over_seeds"] < 0.05,
        "above the null's 97.5th percentile": t > null["p97.5"],
        "beats shuffled-pair map by > 1 std": t - summary[METHODS[3]]["mean"] > max(sd, summary[METHODS[3]]["std"]),
        "beats difficulty baseline (A probe on A) by > 1 std":
            t - summary[METHODS[6]]["mean"] > max(sd, summary[METHODS[6]]["std"]),
    }
    if all(list(checks.values())[:3]):
        v = ("TRANSFERS, beyond shared question difficulty" if checks[list(checks)[3]]
             else "TRANSFERS, but no better than shared question difficulty")
    else:
        v = "DOES NOT TRANSFER (indistinguishable from controls)"
    return {"best_method": best_k, "transfer_auroc": t, "checks": checks, "verdict": v,
            "fraction_of_native": t / summary[METHODS[0]]["mean"]}


def paired_bootstrap(y, s1, s2, n=1000, seed=0):
    rng, d = np.random.default_rng(seed), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            d.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(np.mean(d)), float(lo), float(hi)


def run_pair(A, B, seeds, pca_r=256):
    per_seed, null_all, pvals = [], [], []
    for s in seeds:
        aucs, scores, y = transfer_one_seed(A, B, s, pca_r)
        per_seed.append(aucs)
        null = np.array(aucs.pop("_null"))
        null_all += null.tolist()
        best_s = max(aucs[METHODS[1]], aucs[METHODS[2]])
        # how far from 0.5 could a meaningless rotation get? transfer must beat that, two-sided
        pvals.append(float((1 + (np.abs(null - .5) >= abs(best_s - .5)).sum()) / (1 + len(null))))
        if s == seeds[0]:
            best = max(METHODS[1:3], key=lambda k: aucs[k])
            tests = {
                f"{best} - control: shuffled-pair ridge": paired_bootstrap(y, scores[best], scores[METHODS[3]]),
                f"{best} - difficulty: A probe on A": paired_bootstrap(y, scores[best], scores[METHODS[6]]),
                f"{best} - B min token log-prob": paired_bootstrap(y, scores[best], scores[METHODS[8]]),
            }
    summary = {k: {"mean": float(np.nanmean([p[k] for p in per_seed])),
                   "std": float(np.nanstd([p[k] for p in per_seed]))} for k in METHODS}
    na = np.array(null_all)
    null = {"n": len(na), "mean": float(na.mean()), "p2.5": float(np.percentile(na, 2.5)),
            "p97.5": float(np.percentile(na, 97.5)), "max_dist_from_0.5": float(np.abs(na - .5).max()),
            "perm_p_value_max_over_seeds": max(pvals)}
    return summary, tests, null


# ------------------------------------------------------------------- CKA ---
def centered_gram(X):
    X = X - X.mean(0)
    K = X @ X.T
    return K - K.mean(0) - K.mean(1)[:, None] + K.mean()


def cka_matrix(XA_layers, XB_layers):
    """Linear CKA between every layer of A and every layer of B (Gram form: cheap for wide d)."""
    GA = [centered_gram(XA_layers[:, i].astype(np.float64)) for i in range(XA_layers.shape[1])]
    GB = [centered_gram(XB_layers[:, j].astype(np.float64)) for j in range(XB_layers.shape[1])]
    nA = [np.linalg.norm(g) for g in GA]
    nB = [np.linalg.norm(g) for g in GB]
    return np.array([[float((ga * gb).sum() / (na * nb)) for gb, nb in zip(GB, nB)]
                     for ga, na in zip(GA, nA)])


def sweep_pair(dA, dB, infoA, infoB, pooling, n_cka=800, seed=0, pca_r=256):
    """Map every layer of B onto A's best layer; relate transfer success to CKA."""
    LA, nB = infoA["best_layer"], infoB["n_layers"]
    A_all = load_feats(dA, pooling, list(range(infoA["n_layers"] + 1)))
    B_all = load_feats(dB, pooling, list(range(nB + 1)))
    A_all, B_all = pair(A_all, B_all)
    sub = np.random.default_rng(seed).permutation(len(A_all["idx"]))[:n_cka]
    cka = cka_matrix(A_all["X"][sub], B_all["X"][sub])

    rows = []
    for lb in range(1, nB + 1):
        A1 = {**A_all, "X": A_all["X"][:, LA]}
        B1 = {**B_all, "X": B_all["X"][:, lb]}
        aucs, _, _ = transfer_one_seed(A1, B1, seed, pca_r)
        rows.append({"layer_b": lb, "rel_depth_b": lb / nB, "cka_with_A_best": float(cka[LA, lb]),
                     "transfer": max(aucs[METHODS[1]], aucs[METHODS[2]]),
                     "native": aucs[METHODS[0]], "random": aucs[METHODS[4]]})
        print(f"    B layer {lb:3d}  CKA {cka[LA, lb]:.3f}  transfer {rows[-1]['transfer']:.4f}  "
              f"native {rows[-1]['native']:.4f}")
    r = np.corrcoef([x["cka_with_A_best"] for x in rows], [x["transfer"] for x in rows])[0, 1]
    return {"cka": cka.tolist(), "layer_rows": rows, "cka_transfer_corr": float(r)}


# ----------------------------------------------------------------- plots ---
def plot_matrix(names, M, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    short = [n.replace("-Instruct", "").replace("-it", "") for n in names]
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2.5, 1.3 * len(names) + 2), dpi=160)
    im = ax.imshow(M, cmap="Blues", vmin=0.5, vmax=max(0.9, np.nanmax(M)))
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=11,
                    color="white" if M[i, j] > 0.75 else "#14355E")
    ax.set_xticks(range(len(names)), short, rotation=20, ha="right")
    ax.set_yticks(range(len(names)), short)
    ax.set_xlabel("evaluated on model B (its own errors)"); ax.set_ylabel("probe trained on model A")
    ax.set_title("Cross-model probe transfer (AUROC)\ndiagonal = native probe")
    fig.colorbar(im, fraction=.046); fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_controls(summary, title, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = METHODS[::-1]
    means = [summary[k]["mean"] for k in names]
    stds = [summary[k]["std"] for k in names]
    colors = ["#888" if k.startswith(("control", "B min")) else "#A4551B" if k.startswith("difficulty")
              else "#1E6B4A" if k.startswith("transfer") else "#14355E" for k in names]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160)
    ax.barh(range(len(names)), means, xerr=stds, color=colors, alpha=.9)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    ax.axvline(0.5, ls="--", c="#999", lw=1)
    for i, (m, sd) in enumerate(zip(means, stds)):
        ax.text(m + sd + 0.012, i, f"{m:.3f}", va="center", fontsize=9)
    ax.set_xlim(0.4, 1.0); ax.set_xlabel("AUROC vs model B's own errors")
    ax.set_title(title, fontsize=11); ax.grid(axis="x", alpha=.25)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_sweep(sw, title, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sw["layer_rows"]
    x = [100 * r["rel_depth_b"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    ax.plot(x, [r["native"] for r in rows], c="#14355E", lw=1.6, label="native probe on B")
    ax.plot(x, [r["transfer"] for r in rows], c="#1E6B4A", lw=2.2, marker="o", ms=3, label="transferred from A")
    ax.plot(x, [r["random"] for r in rows], c="#888", lw=1, ls=":", label="random-rotation control")
    ax2 = ax.twinx()
    ax2.plot(x, [r["cka_with_A_best"] for r in rows], c="#A4551B", lw=1.2, ls="--", label="CKA with A's best layer")
    ax2.set_ylabel("linear CKA", color="#A4551B")
    ax.set_xlabel("model B layer (relative depth, %)"); ax.set_ylabel("AUROC")
    ax.set_title(f"{title}   (corr(CKA, transfer) = {sw['cka_transfer_corr']:+.2f})", fontsize=10)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


# ------------------------------------------------------------------ main ---
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder holding one sub-folder per model (notebook 02 output)")
    ap.add_argument("--models", nargs="*", help="sub-folder names; default = every model with a cache")
    ap.add_argument("--pooling", default="a_mean", choices=POOLINGS)
    ap.add_argument("--layer", nargs="*", type=int,
                    help="one layer per model (same order as --models); default = best_layer in results.json")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--pca", type=int, default=256, help="PCA dims for Procrustes")
    ap.add_argument("--sweep", action="store_true", help="also map every layer of B + CKA (slower)")
    ap.add_argument("--out", default=None, help="output folder (default <root>/transfer)")
    a = ap.parse_args(argv)

    dirs = model_dirs(a.root, a.models)
    if len(dirs) < 2:
        raise SystemExit(f"need feature caches for at least 2 models under {a.root}; found {[d.name for d in dirs]}.\n"
                         "Run notebooks/02_improvements.ipynb again with a different MODEL_ID first.")
    infos = [model_info(d, a.pooling) for d in dirs]
    if a.layer:
        for info, L in zip(infos, a.layer):
            info["best_layer"] = L
    for info in infos:
        if info["best_layer"] is None:
            raise SystemExit(f"{info['name']}: no best_layer in results.json -- pass --layer")
    out = Path(a.out) if a.out else Path(a.root) / "transfer"
    out.mkdir(parents=True, exist_ok=True)

    print(f"pooling={a.pooling}  seeds={a.seeds}  models:")
    for i in infos:
        print(f"  {i['name']:32s} layer {i['best_layer']} of {i['n_layers']}")
    feats = {i["name"]: load_feats(d, a.pooling, i["best_layer"]) for d, i in zip(dirs, infos)}

    names = [i["name"] for i in infos]
    K = len(names)
    matrix = np.full((K, K), np.nan)
    report = {"pooling": a.pooling, "seeds": a.seeds, "models": infos, "pairs": {}}
    seeds = list(range(a.seeds))

    for (i, na), (j, nb) in permutations(enumerate(names), 2):
        A, B = pair(feats[na], feats[nb])
        agree = float((A["label"] == B["label"]).mean())
        print(f"\n=== probe from {na}  ->  evaluated on {nb}   ({len(A['idx'])} shared questions, "
              f"labels agree {100 * agree:.0f}%)")
        summary, tests, null = run_pair(A, B, seeds, a.pca)
        for k in METHODS:
            print(f"  {k:42s} {summary[k]['mean']:.4f} +/- {summary[k]['std']:.4f}")
        print(f"  random-rotation null ({null['n']} draws): 95% range [{null['p2.5']:.4f}, {null['p97.5']:.4f}]"
              f"   permutation p <= {null['perm_p_value_max_over_seeds']:.3f}")
        vd = verdict(summary, null)
        for c, ok in vd["checks"].items():
            print(f"  [{'x' if ok else ' '}] {c}")
        print(f"  VERDICT: {vd['verdict']}  ({100 * vd['fraction_of_native']:.0f}% of native)")
        print("  paired bootstrap, seed 0 only (descriptive):")
        for k, (m, lo, hi) in tests.items():
            print(f"    {k:62s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}] {'*' if lo > 0 else ''}")
        matrix[i, j] = vd["transfer_auroc"]
        matrix[j, j] = summary[METHODS[0]]["mean"]
        key = f"{na} -> {nb}"
        report["pairs"][key] = {"n_shared": len(A["idx"]), "label_agreement": agree,
                                "summary": summary, "bootstrap": tests, "random_rotation_null": null,
                                "verdict": vd}
        plot_controls(summary, f"probe from {na} → errors of {nb}", out / f"controls_{na}__to__{nb}.png")

        if a.sweep:
            print("  layer sweep + CKA ...")
            sw = sweep_pair(dirs[i], dirs[j], infos[i], infos[j], a.pooling, pca_r=a.pca)
            report["pairs"][key]["sweep"] = sw
            plot_sweep(sw, f"{na} → {nb}", out / f"sweep_{na}__to__{nb}.png")

    report["names"], report["matrix"] = names, matrix.tolist()
    plot_matrix(names, matrix, out / "transfer_matrix.png")
    (out / "transfer_results.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {out}/transfer_results.json, transfer_matrix.png, controls_*.png")

    print("\nHOW TO READ THIS")
    print("  transfer >> random/shuffled/unaligned controls  -> the alignment carries real correspondence")
    print("  transfer >  'difficulty: A probe on A'          -> B's own state adds signal beyond shared question difficulty")
    print("  transfer close to native                        -> the truthfulness direction is largely shared")
    return report


if __name__ == "__main__":
    main()

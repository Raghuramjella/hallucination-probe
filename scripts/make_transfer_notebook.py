"""Regenerate notebooks/04_cross_model_transfer.ipynb with the current src/transfer.py embedded.

    python scripts/make_transfer_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = (ROOT / "src" / "transfer.py").read_text()


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Hallucination Probe - Part 4: Cross-Model Transfer (RQ2)

**Question:** does a truthfulness probe trained on one model detect the hallucinations of a *different* model?

Model activation spaces are not directly comparable, so we fit a linear map from model B to model A on the
same questions, then score B's mapped activations with A's probe, **against B's own errors**.

| Result | What it would mean |
|---|---|
| transfer far above the random-rotation null | the alignment carries real correspondence |
| transfer above *"A probe on A"* | B's own internal state adds signal beyond "this question is hard" |
| transfer close to native | the truthfulness direction is largely shared across models |

**Before running:** run `02_improvements.ipynb` for **at least two models** with the **same `N_TRIVIA`**
(change `MODEL_ID`, Run all). Each model gets its own folder in `MyDrive/hallucination_probe_v2/`.

CPU only - no GPU needed. About 1-3 min per model pair, plus ~5 min per pair for the layer sweep.
"""),
    code("""
!pip install -q scikit-learn matplotlib
from google.colab import drive
drive.mount("/content/drive")
ROOT = "/content/drive/MyDrive/hallucination_probe_v2"
"""),
    md("The transfer code (identical to `src/transfer.py` in the repo):"),
    code("%%writefile transfer.py\n" + SRC),
    code("""
from pathlib import Path
import transfer
dirs = transfer.model_dirs(ROOT)
for d in dirs:
    info = transfer.model_info(d)
    n = len(list(d.glob("trivia_feats_*.npz")))
    print(f"{info['name']:34s} best layer {info['best_layer']} of {info['n_layers']}   ({n} shards)")
assert len(dirs) >= 2, "need at least two models - run notebook 02 with another MODEL_ID first"
"""),
    md("""
## Run the transfer study

`pooling="a_mean"` matches the main results. Try `"p_last"` too: the last prompt token is the same
question text for both models, so it is the cleanest pairing.
"""),
    code("""
report = transfer.main(["--root", ROOT, "--pooling", "a_mean", "--seeds", "5"])
"""),
    code("""
from IPython.display import Image, display
OUT = Path(ROOT) / "transfer"
display(Image(str(OUT / "transfer_matrix.png"), width=520))
for p in sorted(OUT.glob("controls_*.png")):
    display(Image(str(p), width=760))
"""),
    md("## Summary table for the report"),
    code("""
M = transfer.METHODS
print(f"{'pair':58s} {'native':>7s} {'transfer':>9s} {'null 97.5%':>10s} {'difficulty':>10s}  verdict")
for pair, r in report["pairs"].items():
    s, v, nl = r["summary"], r["verdict"], r["random_rotation_null"]
    print(f"{pair:58s} {s[M[0]]['mean']:7.3f} {v['transfer_auroc']:9.3f} {nl['p97.5']:10.3f} "
          f"{s[M[6]]['mean']:10.3f}  {v['verdict']}")
"""),
    md("""
## Optional: layer sweep + CKA (the "why" figure)

Maps **every** layer of model B onto model A's best layer, and plots transfer success against linear CKA
similarity. If CKA predicts transfer, the result has an explanation, not just a number.
"""),
    code("""
report_sweep = transfer.main(["--root", ROOT, "--pooling", "a_mean", "--seeds", "3", "--sweep",
                              "--out", str(Path(ROOT) / "transfer_sweep")])
for p in sorted((Path(ROOT) / "transfer_sweep").glob("sweep_*.png")):
    display(Image(str(p), width=760))
"""),
]

nb = {"cells": cells, "metadata": {"colab": {"provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 0}
out = ROOT / "notebooks" / "04_cross_model_transfer.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)

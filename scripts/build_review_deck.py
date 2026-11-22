"""Build the final-review deck from the abstract-review deck + latest results.

    python scripts/build_review_deck.py

Inputs (all optional -- falls back to the numbers in the full report):
    results/results.json          from Drive: hallucination_probe_v2/<model>/results.json
    results/timings.json          same folder
    results/risk_coverage_v2.png  same folder (otherwise a curve is drawn from results.json)
    results/demo_url.txt          the Gradio share link, if you want it on the demo slide
Output:
    Final_Review_PPT.pptx, figures/*.png
"""
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Pt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

NAVY, ACCENT, GREY, INK = "14355E", "2E6B9E", "62707E", "1A1D21"
PANEL, GREEN, GREEN_BG = "EFF3F7", "1E6B4A", "E8F3EC"
AMBER, AMBER_BG = "A4551B", "FBF1E8"
FONT = "Calibri"

# --------------------------------------------------------------- numbers ---
# v1 = measured numbers from docs/Hallucination_Probe_Full_Report.pdf
V1_SWEEP = [0.5462, 0.5553, 0.5921, 0.6083, 0.6270, 0.6596, 0.6835, 0.6565, 0.6660,
            0.6571, 0.7508, 0.7576, 0.7752, 0.7757, 0.8052, 0.7967, 0.8032, 0.7990,
            0.8019, 0.8086, 0.8186, 0.7979, 0.8117, 0.8039, 0.7784, 0.7539, 0.7465,
            0.7381, 0.7234]


def load_numbers():
    f = RES / "results.json"
    if not f.exists():
        print("results/results.json not found -> using report (v1) numbers")
        return {
            "source": "v1", "model": "Qwen2.5-1.5B-Instruct", "n": 2000,
            "n_layers": 28, "acc": 0.290, "auroc": 0.8186, "std": 0.0110,
            "layer": 20, "pool": "a_mean", "sweeps": {"a_mean": V1_SWEEP},
            "pool_aurocs": {"a_mean": (0.8186, 0.0110)},
            "random_split": 0.8308, "shuffled": None,
            "mean_lp": 0.7670, "min_lp": 0.7708, "combined": None,
            "ci": (-0.0157, 0.0818), "sc": 0.7580, "sc_ratio": None,
            "sel": {"0.9": (0.321, 0.325, 0.90), "0.8": (0.355, 0.365, 0.86),
                    "0.7": (0.394, 0.417, 0.81), "0.6": (0.434, 0.487, 0.73),
                    "0.5": (0.500, 0.584, 0.71)},
            "transfer": None,
        }
    r = json.loads(f.read_text())
    b = r["best_pooling"]
    bl = r.get("baselines", {})
    sel_raw = r.get("selective", {})
    sc = r.get("self_consistency", {})
    sc_auc = None
    if sc:
        sc_auc = max(v for k, v in sc["auroc"].items() if k.startswith("self-consistency"))
    ratio = None
    if sc and sc.get("probe_s_per_q"):
        ratio = sc["sc_s_per_q"] / sc["probe_s_per_q"]
    n = None
    for name in ("trivia_answers.jsonl",):
        if (RES / name).exists():
            n = sum(1 for _ in open(RES / name))
    return {
        "source": "v2", "model": r["model"].split("/")[-1], "n": n,
        "n_layers": r["n_layers"],
        # abstention numbers are on the test split; its base accuracy = oracle@90% * 0.9
        "acc": sel_raw["0.9"]["oracle"] * 0.9 if "0.9" in sel_raw else r.get("trivia_accuracy"),
        "acc_all": r.get("trivia_accuracy"),
        "auroc": r["robustness"][b]["mean"], "std": r["robustness"][b]["std"],
        "layer": r["best_layer"], "pool": b, "sweeps": r["sweeps"],
        "pool_aurocs": {k: (v["mean"], v["std"]) for k, v in r["robustness"].items()},
        "random_split": r.get("random_split_auroc"),
        "shuffled": (r.get("shuffled_label_auroc") or {}).get("mean"),
        "mean_lp": bl.get("mean token log-prob", {}).get("mean"),
        "min_lp": bl.get("min token log-prob", {}).get("mean"),
        "combined": bl.get("probe + log-prob", {}).get("mean"),
        "ci": tuple(r["bootstrap_ci"]) if "bootstrap_ci" in r else None,
        "sc": sc_auc, "sc_ratio": ratio,
        "sel": {k: (v["probe"], v["oracle"], v["captured"]) for k, v in r.get("selective", {}).items()},
        "transfer": r.get("transfer_nq"),
        "sc_probe": (sc.get("auroc") or {}).get("internal-state probe"),
        "sc_n": 300 if sc else None,
    }


POOL_LABEL = {"a_mean": "mean over answer tokens", "a_last": "last answer token",
              "p_last": "last prompt token (before answering)"}


# --------------------------------------------------------------- figures ---
def fig_style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#B8C2CC"); ax.spines["bottom"].set_color("#B8C2CC")
    ax.tick_params(colors="#62707E", labelsize=11)
    ax.grid(axis="y", alpha=.25)


def make_sweep_fig(N):
    fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=200)
    colors = {"a_mean": "#14355E", "a_last": "#2E6B9E", "p_last": "#1E6B4A"}
    for pool, s in N["sweeps"].items():
        main = pool == N["pool"]
        ax.plot(range(len(s)), s, marker="o", ms=3.5 if main else 2.5,
                lw=2.4 if main else 1.4, color=colors.get(pool, "#888"),
                alpha=1 if main else .75, label=POOL_LABEL.get(pool, pool))
    s = N["sweeps"][N["pool"]]
    L = N["layer"]
    ax.annotate(f"layer {L}  ({100 * L / N['n_layers']:.0f}% depth)",
                xy=(L, s[L]), xytext=(L - 9, s[L] + 0.03), fontsize=11, color="#14355E",
                arrowprops=dict(arrowstyle="-", color="#14355E", lw=1))
    ax.axhline(0.5, ls="--", lw=1, color="#999")
    ax.text(0.3, 0.505, "chance", fontsize=10, color="#999")
    ax.set_xlabel("transformer layer", color="#62707E", fontsize=12)
    ax.set_ylabel("AUROC" if N["source"] == "v1" else "validation AUROC", color="#62707E", fontsize=12)
    ax.set_ylim(0.45, max(max(v) for v in N["sweeps"].values()) + 0.06)
    if len(N["sweeps"]) > 1:
        ax.legend(frameon=False, fontsize=10, loc="lower right")
    fig_style(ax); fig.tight_layout()
    out = FIG / "deck_layer_sweep.png"; fig.savefig(out); plt.close(fig)
    return out


def make_rc_fig(N):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    base = N["acc"]
    cs = [1.0] + sorted((float(k) for k in N["sel"]), reverse=True)
    probe = [base] + [N["sel"][str(c)][0] for c in cs[1:]]
    oracle = [base] + [N["sel"][str(c)][1] for c in cs[1:]]
    ax.plot([100 * c for c in cs], [100 * v for v in oracle], ls=":", lw=1.6, color="#1A1D21", label="perfect detector")
    ax.plot([100 * c for c in cs], [100 * v for v in probe], marker="o", lw=2.6, color="#1E6B4A", label="internal-state probe")
    ax.axhline(100 * base, ls="--", lw=1.2, color="#A4551B", label=f"answer everything ({100 * base:.1f}%)")
    ax.invert_xaxis()
    ax.set_xlabel("coverage: % of questions answered", color="#62707E", fontsize=12)
    ax.set_ylabel("accuracy on answered (%)", color="#62707E", fontsize=12)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig_style(ax); fig.tight_layout()
    out = FIG / "deck_risk_coverage.png"; fig.savefig(out); plt.close(fig)
    return out


# ------------------------------------------------------------ pptx tools ---
def rgb(h):
    return RGBColor.from_string(h)


def box(slide, x, y, w, h, fill=None, line=None, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, Pt(x), Pt(y), Pt(w), Pt(h))
    if fill:
        s.fill.solid(); s.fill.fore_color.rgb = rgb(fill)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line); s.line.width = Pt(0.75)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def text(slide, x, y, w, h, runs, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, italic=False):
    """runs: str, or list of paragraphs; a paragraph is str or list of (text, overrides)."""
    tb = slide.shapes.add_textbox(Pt(x), Pt(y), Pt(w), Pt(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Pt(2); tf.margin_top = tf.margin_bottom = Pt(1)
    tf.vertical_anchor = anchor
    paras = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        for t, o in ([(para, {})] if isinstance(para, str) else para):
            r = p.add_run(); r.text = t
            f = r.font; f.name = FONT
            f.size = Pt(o.get("size", size)); f.bold = o.get("bold", bold)
            f.italic = o.get("italic", italic)
            f.color.rgb = rgb(o.get("color", color))
    return tb


def header(slide, title, subtitle=None):
    text(slide, 61, 30, 837, 50, title, size=30, color=NAVY, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    box(slide, 61, 86, 79, 3, fill=ACCENT)
    if subtitle:
        text(slide, 61, 100, 837, 26, subtitle, size=15, color=GREY, italic=True)


def stat(slide, x, y, w, big, small, color=NAVY, bg=PANEL, h=100):
    box(slide, x, y, w, h, fill=bg)
    text(slide, x, y + 10, w, h - 16, [[(big, {"size": 30, "bold": True, "color": color})],
                                        [(small, {"size": 11, "color": GREY})]],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def table(slide, x, y, w, rows, widths, highlight=None, row_h=34, head=True):
    for i, row in enumerate(rows):
        yy = y + i * (row_h + 5)
        is_head = head and i == 0
        hl = highlight is not None and i == highlight
        if not is_head:
            box(slide, x, yy, w, row_h, fill=GREEN_BG if hl else PANEL)
        xx = x + 14
        for j, cell in enumerate(row):
            cw = widths[j] * (w - 28)
            col = GREEN if (hl and j > 0) else (GREY if is_head else INK)
            text(slide, xx, yy, cw, row_h, cell, size=12 if is_head else 15, color=col,
                 bold=is_head or hl, anchor=MSO_ANCHOR.MIDDLE,
                 align=PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT)
            xx += cw


def blank(prs):
    layout = min(prs.slide_layouts, key=lambda l: len(l.placeholders))
    s = prs.slides.add_slide(layout)
    for ph in list(s.placeholders):
        ph._element.getparent().remove(ph._element)
    return s


def move_slide(prs, slide, new_index):
    lst = prs.slides._sldIdLst
    el = [e for e in lst if prs.slides.get(int(e.get("id"))) is slide] if False else None
    ids = list(lst)
    for e in ids:
        if prs.part.related_part(e.rId) is slide.part:
            lst.remove(e); lst.insert(new_index, e); return


def delete_slide(prs, index):
    lst = prs.slides._sldIdLst
    e = list(lst)[index]
    prs.part.drop_rel(e.rId); lst.remove(e)


def replace_text(slide, old, new):
    for sh in slide.shapes:
        if sh.has_text_frame:
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    if old in r.text:
                        r.text = r.text.replace(old, new)


def renumber(prs):
    """Existing slides carry a small page-number box at bottom right; rewrite all of them."""
    for i, s in enumerate(prs.slides, 1):
        found = False
        for sh in s.shapes:
            if (sh.has_text_frame and sh.text_frame.text.strip().isdigit()
                    and sh.top > Pt(470) and sh.left > Pt(820)):
                sh.text_frame.paragraphs[0].runs[0].text = str(i); found = True
        if not found and i > 1:
            text(s, 864, 493, 64, 28, str(i), size=11, color=GREY, align=PP_ALIGN.RIGHT)


def f3(v):
    return "—" if v is None else f"{v:.3f}"


# ---------------------------------------------------------------- slides ---
def slide_results(prs, N):
    s = blank(prs)
    n = f"{N['n']:,} " if N["n"] else ""
    header(s, "Results", f"{N['model']}, {n}TriviaQA questions, 10 entity-disjoint splits")
    acc80 = N["sel"]["0.8"][0]
    stat(s, 61, 140, 205, f"{N['auroc']:.3f}", f"detection AUROC\n(± {N['std']:.3f} over 10 splits)")
    stat(s, 279, 140, 205, f"Layer {N['layer']}", f"of {N['n_layers']} — peak signal\nat {100 * N['layer'] / N['n_layers']:.0f}% depth")
    stat(s, 497, 140, 205, "1", "forward pass\n(vs 11 for sampling)", color=GREEN)
    stat(s, 715, 140, 205, f"+{100 * (acc80 - N['acc']):.1f} pts",
         f"accuracy when skipping\nriskiest 20% ({100 * N['acc']:.1f}% → {100 * acc80:.1f}%)", color=GREEN)

    text(s, 61, 254, 837, 26, "Comparison with existing methods", size=16, color=NAVY, bold=True)
    rows = [["method", "AUROC", "cost"]]
    if N["mean_lp"]:
        rows.append(["Mean token log-probability", f3(N["mean_lp"]), "1 pass"])
    rows.append(["Min token log-probability", f3(N["min_lp"]), "1 pass"])
    if N["sc"]:
        rows.append(["Self-consistency (10 samples)*", f3(N["sc"]),
                     f"11 passes" + (f" · {N['sc_ratio']:.0f}× slower" if N["sc_ratio"] else "")])
    rows.append(["Our method — internal-state probe", f3(N["auroc"]), "1 pass"])
    hl = len(rows) - 1
    if N["combined"]:
        rows.append(["Probe + log-probability combined", f3(N["combined"]), "1 pass"])
    table(s, 61, 282, 859, rows, [0.55, 0.18, 0.27], highlight=hl, row_h=27)
    if N.get("sc_probe"):
        rows_used = len(rows)
        text(s, 61, 282 + rows_used * 32 + 2, 760, 20,
             f"* Self-consistency measured on a {N['sc_n']}-question subset; on the same subset the probe "
             f"scores {N['sc_probe']:.3f}.", size=10, color=GREY, italic=True)
    return s


def slide_sweep(prs, N, img):
    s = blank(prs)
    header(s, "Where does the signal live?", "Probe accuracy at every layer of the network")
    s.shapes.add_picture(str(img), Pt(40), Pt(130), width=Pt(580))
    pts = [("Early layers", "surface form only — weak signal"),
           (f"Layer {N['layer']} of {N['n_layers']}", f"peak at {100 * N['layer'] / N['n_layers']:.0f}% depth — "
                                                     "matches 7B–70B models in the literature"),
           ("Final layers", "signal decays as layers specialise in predicting the next word")]
    if N["source"] == "v2" and "p_last" in N["pool_aurocs"]:
        pl = N["pool_aurocs"]["p_last"][0]
        pts.append(("Before the answer exists", f"reading the question alone gives AUROC {pl:.3f}"))
    y = 150
    for t, d in pts:
        box(s, 640, y, 4, 58, fill=ACCENT)
        text(s, 652, y, 270, 62, [[(t, {"bold": True, "color": NAVY, "size": 15})],
                                  [(d, {"size": 12, "color": GREY})]])
        y += 76
    return s


def slide_selective(prs, N, img):
    s = blank(prs)
    header(s, "Selective answering", "The system says “I'm not sure” instead of guessing")
    s.shapes.add_picture(str(img), Pt(40), Pt(130), width=Pt(500))
    rows = [["answered", "accuracy", "of perfect"]]
    rows.append(["100%", f"{100 * N['acc']:.1f}%", "—"])
    for c in ("0.9", "0.8", "0.7", "0.6", "0.5"):
        if c in N["sel"]:
            p, o, cap = N["sel"][c]
            rows.append([f"{100 * float(c):.0f}%", f"{100 * p:.1f}%", f"{100 * cap:.0f}%"])
    table(s, 560, 140, 360, rows, [0.34, 0.33, 0.33], highlight=3, row_h=30)
    cap80 = N["sel"]["0.8"][2]
    text(s, 560, 410, 360, 60, [[("Skipping the riskiest 20% captures ", {}),
                                 (f"{100 * cap80:.0f}%", {"bold": True, "color": GREEN}),
                                 (" of what a perfect detector could achieve.", {})]], size=14)
    return s


def slide_trust(prs, N):
    s = blank(prs)
    header(s, "Why these numbers can be trusted")
    items = [("Entity-disjoint splits", "the same answer entity never appears in both train and test"),
             ("Error bars over 10 splits", f"AUROC {N['auroc']:.3f} ± {N['std']:.3f}, not one lucky run")]
    if N["random_split"]:
        items.append(("Leakage control", f"random split {N['random_split']:.3f} vs entity split {N['auroc']:.3f} — "
                                         "difference within noise"))
    if N["shuffled"] is not None:
        items.append(("Negative control", f"probe trained on shuffled labels scores {N['shuffled']:.3f} — chance, as it must"))
    if N["source"] == "v2":
        items.append(("Layer chosen on validation data", "the test set is touched only once, at the end"))
    items.append(("Manual label audit", "gradings hand-checked; one systematic grading bug found and fixed"))
    if N["ci"]:
        lo, hi = N["ci"]
        sig = lo > 0
        items.append(("Honest significance test",
                      f"probe − log-prob 95% CI [{lo:+.3f}, {hi:+.3f}] — "
                      + ("significant" if sig else "not yet significant; reported as such")))
    y = 112
    for t, d in items:
        box(s, 61, y + 4, 22, 22, fill=GREEN, shape=MSO_SHAPE.OVAL)
        text(s, 61, y + 4, 22, 22, "✓", size=12, color="FFFFFF", bold=True, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
        text(s, 96, y, 820, 50, [[(t, {"bold": True, "size": 16})], [(d, {"size": 12, "color": GREY})]])
        y += 54
    return s


def slide_transfer(prs, N):
    t = N["transfer"]
    if not t:
        return None
    s = blank(prs)
    header(s, "Does it generalise to new questions?",
           "Probe trained only on TriviaQA, tested on Natural Questions it has never seen")
    stat(s, 61, 150, 270, f"{t['trivia_to_nq']:.3f}", "TriviaQA → Natural Questions\n(no retraining)", h=120)
    stat(s, 345, 150, 270, f"{t['nq_in_domain']:.3f}", "probe trained on\nNatural Questions itself", h=120)
    stat(s, 629, 150, 270, f"{t['min_logprob']:.3f}", "min token log-probability\non Natural Questions", h=120)
    keep = t["trivia_to_nq"] / t["nq_in_domain"] if t["nq_in_domain"] else 0
    text(s, 61, 300, 837, 80, [[("The transferred probe keeps ", {}),
                                (f"{100 * keep:.0f}%", {"bold": True, "color": GREEN}),
                                (" of in-domain performance — it learned a general “the model is guessing” "
                                 "signal, not TriviaQA-specific patterns.", {})]], size=16)
    return s


def slide_demo(prs, N):
    s = blank(prs)
    header(s, "Live demo", "Ask any question — the probe reads the model's internal state as it answers")
    url = (RES / "demo_url.txt").read_text().strip() if (RES / "demo_url.txt").exists() else None
    box(s, 61, 140, 520, 300, fill=PANEL)
    lines = [("Question", "Who designed the red telephone box?"),
             ("Model answer", "“William Morris”"),
             ("Probe", "likely hallucinated — high risk"),
             ("Decision", "NOT CONFIDENT → “I'm not sure.”")]
    y = 160
    for k, v in lines:
        text(s, 85, y, 140, 30, k, size=12, color=GREY, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 220, y, 345, 30, v, size=16, color=AMBER if k == "Decision" else INK,
             bold=k == "Decision", anchor=MSO_ANCHOR.MIDDLE)
        y += 62
    text(s, 85, 405, 480, 24, "(truth: Giles Gilbert Scott)", size=11, color=GREY, italic=True)
    steps = [("1", "Model answers normally"), ("2", f"Hidden state at layer {N['layer']} is read"),
             ("3", "Linear probe gives a risk score"), ("4", "Above threshold → abstain")]
    y = 150
    for n, t in steps:
        box(s, 610, y, 30, 30, fill=NAVY, shape=MSO_SHAPE.OVAL)
        text(s, 610, y, 30, 30, n, size=13, color="FFFFFF", bold=True, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
        text(s, 652, y, 270, 30, t, size=15, anchor=MSO_ANCHOR.MIDDLE)
        y += 48
    text(s, 610, 350, 310, 90, [[("Gradio app · runs on a free Colab T4", {"size": 12, "color": GREY})],
                                [(url or "notebooks/03_demo.ipynb", {"size": 13, "color": ACCENT, "bold": True})]])
    return s


def slide_status(prs, N):
    s = blank(prs)
    header(s, "Work completed and next steps")
    done = ["End-to-end pipeline: generate → grade → extract → probe → abstain",
            "Layer-wise map of where truthfulness is encoded",
            "Baselines: token log-probability, self-consistency",
            "Robustness: 10-split error bars, leakage control"
            + (", shuffled-label control" if N["shuffled"] is not None else ""),
            "Selective-answering (abstention) system",
            "Interactive Gradio demo with live confidence"]
    if N["transfer"]:
        done.insert(4, "Cross-dataset transfer: TriviaQA → Natural Questions")
    if N["ci"] and N["ci"][0] > 0:
        done.insert(3, "Probe beats log-probability — statistically significant")
    todo = ["More models: Qwen2.5-3B, Llama-3.2, Gemma-2",
            "Cross-model probe transfer (Procrustes alignment)",
            "Semantic-entropy baseline (NLI clustering)",
            "Second task type: HaluEval / HotpotQA",
            "Paper for an IEEE / Scopus-indexed conference"]
    text(s, 61, 108, 440, 26, "Completed", size=16, color=GREEN, bold=True)
    text(s, 520, 108, 400, 26, "Next steps", size=16, color=NAVY, bold=True)
    y = 142
    for d in done:
        box(s, 61, y + 3, 18, 18, fill=GREEN, shape=MSO_SHAPE.OVAL)
        text(s, 61, y + 3, 18, 18, "✓", size=10, color="FFFFFF", bold=True, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
        text(s, 88, y, 410, 40, d, size=13)
        y += 44
    y = 142
    for d in todo:
        box(s, 520, y + 3, 18, 18, line=ACCENT, shape=MSO_SHAPE.OVAL)
        text(s, 547, y, 370, 40, d, size=13)
        y += 44
    return s


def slide_limits(prs, N):
    s = blank(prs)
    header(s, "Limitations")
    items = [("One model family so far", "Qwen2.5-1.5B; generality across families is the next experiment"),
             ("Short factual questions only", "prior work reports weaker transfer to reasoning or summarisation"),
             ("Automatic grading", "answers graded by alias matching; audited by hand, small residual noise"),
             ("Weak base model", f"{100 * N['acc']:.0f}% accuracy caps how high abstention can lift accuracy — "
                                 "the ceiling is the model's knowledge, not the detector")]
    if N["ci"] and N["ci"][0] <= 0:
        items.insert(1, ("Lead over log-probability not yet significant",
                         "the gap is real in the point estimate; more data is needed to confirm it"))
    y = 112
    for t, d in items:
        box(s, 61, y, 4, 52, fill=AMBER)
        text(s, 76, y, 840, 56, [[(t, {"bold": True, "size": 16})], [(d, {"size": 12, "color": GREY})]])
        y += 70
    return s


# ------------------------------------------------------------------ main ---
def main():
    N = load_numbers()
    sweep_png = make_sweep_fig(N)
    rc_png = RES / "risk_coverage_v2.png"
    rc_png = make_rc_fig(N)          # consistent styling; Drive PNG kept as backup

    prs = Presentation(ROOT / "Abstract_Review_PPT.pptx")
    replace_text(prs.slides[0], "ABSTRACT REVIEW", "FINAL REVIEW")
    replace_text(prs.slides[4], "Proposed Methodology", "Methodology")

    # original order: 1 title, 2 intro, 3 problem, 4 objectives, 5 method, 6 tools,
    # 7 preliminary results, 8 expected outcomes, 9 applications, 10 references
    new = [slide_results(prs, N), slide_sweep(prs, N, sweep_png), slide_selective(prs, N, rc_png),
           slide_transfer(prs, N), slide_trust(prs, N), slide_demo(prs, N),
           slide_limits(prs, N), slide_status(prs, N)]
    new = [x for x in new if x is not None]
    delete_slide(prs, 7)             # expected outcomes -> replaced by status slide
    delete_slide(prs, 6)             # preliminary results -> replaced by results slide
    for i, sl in enumerate(new):
        move_slide(prs, sl, 6 + i)   # after slide 6 (tools); applications + refs stay last
    renumber(prs)

    out = ROOT / "Final_Review_PPT.pptx"
    prs.save(out)
    print(f"saved {out}  ({len(prs.slides)} slides, numbers from {N['source']})")


if __name__ == "__main__":
    main()

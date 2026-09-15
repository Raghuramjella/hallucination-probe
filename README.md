# Hallucination Probe

Detecting when a language model is about to be wrong, by reading its own
internal activations instead of sampling it ten times.

```
Question --> [ MODEL ] --> Answer
                 |
          internal numbers
                 |
          [ tiny probe ] --> "this is probably wrong"
```

The model already computes thousands of intermediate numbers before it emits a
word. We capture those, and train a logistic regression to spot the pattern that
appears when the model is guessing. Cost: one dot product on numbers we already
had. Compare with self-consistency methods, which cost ten extra generations.

---

## Fastest path: the Colab notebook

Upload `notebooks/01_hallucination_probe_colab.ipynb` to Google Colab,
set **Runtime -> Change runtime type -> T4 GPU**, and run the cells in order.
Self-contained, saves to Drive, resumable after a disconnect.

**You will know whether the idea works by the end of Stage 3 - about two hours.**

## Or run the scripts

```bash
pip install -r requirements.txt

python src/generate.py     # Stage 1: ask 2000 questions, grade answers   (GPU, ~30 min)
python src/audit.py 30     # eyeball the grading BEFORE trusting anything
python src/extract.py      # Stage 2: capture activations, shard to disk  (GPU, ~15 min)
python src/probe.py        # Stages 3-4: train probe + sweep every layer  (CPU, seconds)
python src/selective.py    # Stage 5: the abstention number that sells it (CPU, seconds)
```

Every stage is resumable. Settings live in `src/config.py` - change them there.

---

## Reading the result

`probe.py` prints an AUROC per layer. Interpretation:

| AUROC | Meaning |
|---|---|
| ~0.50 | Chance. Check the labels, or try a 3B model |
| 0.65 - 0.75 | Real signal. You are in business |
| 0.75 - 0.85 | Strong. A genuinely usable detector |
| above 0.95 | **Suspicious.** Assume a leak until proven otherwise |

Expect the curve to peak somewhere around 60-80% of network depth. Early layers
handle surface form; the last layers specialise in next-token prediction; meaning
lives in the middle.

---

## Three design decisions worth understanding

**1. Two-pass extraction.** Generation happens first (`generate.py`), then a
separate forward pass over prompt+answer captures hidden states
(`extract.py`). Extracting during generation means wrestling a tuple nested over
both decode steps and layers. Two passes is slightly slower and far easier to
get right.

**2. Extract once, experiment forever.** The activation shards are the whole
point. After `extract.py` finishes you never touch the GPU again - every probe,
every layer sweep, every ablation is a CPU operation taking seconds. This is
what makes the project affordable on a free Colab tier.

**3. Entity-grouped splits, never random.** If "Eiffel Tower" appears in both
train and test, the probe learns *entity frequency* rather than *truthfulness*,
and AUROC inflates while telling you nothing. `probe.py` splits by answer entity
and also prints what a random split would have scored - that gap is the leak you
avoided.

---

## Known weak points

- **Label noise.** Correctness is graded by substring match against gold
  aliases. "Paris, France" vs "Paris" grades fine; short or common aliases
  misgrade. Run `audit.py`, count your disagreements, and report the rate.
- **Small model.** At 1.5B the signal may be weak. If AUROC sits near 0.55, try
  `Qwen/Qwen2.5-3B-Instruct` before concluding anything.
- **One dataset.** TriviaQA is short-form factual recall. Conclusions may not
  transfer to reasoning or summarisation.

## Next steps

1. Audit labels, record the noise rate.
2. Add a log-probability baseline, so you can show the probe beats the free option.
3. Sweep pooling: mean over answer tokens vs. last token only.
4. Second model - does the same layer win? Does the probe transfer?
5. Gradio demo with a live confidence bar.

## Layout

```
src/config.py        all settings
src/grading.py       answer normalisation + correctness
src/model_utils.py   model loading + prompt construction (shared - do not duplicate)
src/generate.py      Stage 1
src/extract.py       Stage 2
src/probe.py         Stages 3-4
src/selective.py     Stage 5
src/audit.py         manual label check
notebooks/           self-contained Colab version
data/                answers.jsonl, acts_*.npy, labels.npy   (gitignored)
artifacts/           trained probe                            (gitignored)
figures/             layer_sweep.png, risk_coverage.png
```

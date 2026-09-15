"""STAGE 1 -- ask the model 2000 questions and grade its answers.

Output: data/answers.jsonl with one record per question:
    {question, prompt, answer, aliases, label, group}
where label = 1 means the model was WRONG (a hallucination).

Resumable: rerun after a disconnect and it picks up where it stopped.
"""
import json
import sys

import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from config import (ANSWERS_FILE, DATASET, DATASET_CONFIG, MAX_NEW_TOKENS,
                    N_EXAMPLES, SPLIT)
from grading import group_key, is_correct
from model_utils import build_prompt, load_model

BATCH = 8


def main():
    tok, model = load_model()
    ds = load_dataset(DATASET, DATASET_CONFIG, split=SPLIT).select(range(N_EXAMPLES))

    done = 0
    if ANSWERS_FILE.exists():
        with open(ANSWERS_FILE) as f:
            done = sum(1 for _ in f)
        print(f"resuming: {done} already generated")

    out = open(ANSWERS_FILE, "a", encoding="utf-8")
    n_wrong = 0

    for start in tqdm(range(done, len(ds), BATCH)):
        rows = [ds[i] for i in range(start, min(start + BATCH, len(ds)))]
        prompts = [build_prompt(tok, r["question"]) for r in rows]

        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, pad_token_id=tok.pad_token_id)

        new_tokens = gen[:, enc.input_ids.shape[1]:]
        answers = tok.batch_decode(new_tokens, skip_special_tokens=True)

        for row, prompt, answer in zip(rows, prompts, answers):
            aliases = row["answer"]["aliases"] or [row["answer"]["value"]]
            answer = answer.strip()
            wrong = 0 if is_correct(answer, aliases) else 1
            n_wrong += wrong
            out.write(json.dumps({
                "question": row["question"],
                "prompt": prompt,
                "answer": answer,
                "aliases": aliases,
                "label": wrong,
                "group": group_key(aliases),
            }, ensure_ascii=False) + "\n")
        out.flush()

    out.close()
    total = len(ds) - done
    if total > 0:
        print(f"\nwrong on {n_wrong}/{total} of the newly generated "
              f"({100 * n_wrong / total:.1f}% hallucination rate)")
    print("\nSANITY CHECK: a 1.5B model should land near 40-60% accuracy.")
    print("If accuracy is >95% or <5%, the grading is broken -- fix it now.")


if __name__ == "__main__":
    main()

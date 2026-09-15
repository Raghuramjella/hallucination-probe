"""Print a random sample of graded answers for MANUAL checking.

Do this before trusting any AUROC. Count how many gradings you disagree with
and report that number in your write-up -- it bounds every result you have.
"""
import json
import random
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from config import ANSWERS_FILE

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30

rows = [json.loads(l) for l in open(ANSWERS_FILE, encoding="utf-8")]
random.seed(0)
for i, r in enumerate(random.sample(rows, min(N, len(rows))), 1):
    verdict = "WRONG" if r["label"] else "right"
    print(f"\n[{i}] graded: {verdict}")
    print(f"  Q: {r['question']}")
    print(f"  model: {r['answer']!r}")
    print(f"  gold : {r['aliases'][:4]}")
print(f"\n--- count your disagreements out of {min(N, len(rows))}. "
      "That is your label-noise rate. ---")

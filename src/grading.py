"""Answer normalisation and correctness grading.

This is the noisiest part of the pipeline. Everything downstream inherits
its errors, so hand-audit a sample before trusting any result.
"""
import re
import string

_PUNCT = set(string.punctuation)


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _contains(text: str, phrase: str) -> bool:
    return re.search(r"\b" + re.escape(phrase) + r"\b", text) is not None


def is_correct(prediction: str, aliases, question: str = "") -> bool:
    """True if any gold alias appears as a whole word/phrase in the prediction.

    - Word-boundary match, not raw substring: "us" no longer matches "bus",
      so short answers like "8" or "UK" can be graded at all.
    - An alias that already appears in the QUESTION is ignored: otherwise a
      model echoing "the Georgia Peach" is graded right for a nickname question
      (the systematic error found in the label audit).
    """
    p, q = normalize(prediction), normalize(question)
    if not p:
        return False
    for a in aliases:
        na = normalize(a)
        if not na or (q and _contains(q, na)):
            continue
        if _contains(p, na):
            return True
    return False


def group_key(aliases) -> str:
    """Entity used to group examples so splits stay disjoint.

    Without this, the same entity lands in train AND test, the probe learns
    entity frequency instead of truthfulness, and AUROC silently inflates.
    """
    return normalize(aliases[0]) if aliases else "__unknown__"

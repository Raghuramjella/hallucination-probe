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


def is_correct(prediction: str, aliases) -> bool:
    """True if any gold alias appears in the (normalised) prediction.

    Substring containment rather than exact match, because a chat model
    answers "The capital is Paris" where the gold alias is "Paris".
    The cost is false positives on aliases that are common words -- which
    is why very short aliases are ignored.
    """
    p = normalize(prediction)
    if not p:
        return False
    for a in aliases:
        na = normalize(a)
        if len(na) >= 3 and na in p:
            return True
    return False


def group_key(aliases) -> str:
    """Entity used to group examples so splits stay disjoint.

    Without this, the same entity lands in train AND test, the probe learns
    entity frequency instead of truthfulness, and AUROC silently inflates.
    """
    return normalize(aliases[0]) if aliases else "__unknown__"

"""Score completions into 0/1 correctness.

Keep exact-match (deterministic) and LLM-judge (stochastic, separate analysis)
strictly apart — this separation isolates sampling noise (H3) from judge noise (H2).
"""
from __future__ import annotations
import re


def exact_match(prediction: str, gold: str, task: str) -> int:
    """Deterministic scorers for exact-match datasets. Extend per dataset."""
    p, g = prediction.strip(), str(gold).strip()
    if task in ("gsm8k", "math500", "mathbench"):
        return int(_last_number(p) == _last_number(g))
    if task in ("mmlu_pro", "gpqa"):                 # multiple choice letter
        return int(_first_choice(p) == _first_choice(g))
    if task in ("mbpp", "humaneval", "livecodebench"):
        # TODO: run unit tests via a sandboxed executor; placeholder string match
        return int(p == g)
    return int(p == g)


def judge_score(prediction: str, gold: str, judge_model: str) -> int:
    """LLM-as-judge for open-ended datasets. Use a fixed non-self judge + one
    cross-judge for the H2 circularity check. Stub — wire the judge API."""
    raise NotImplementedError("Wire LLM-judge API; record judge_model + seed.")


def _last_number(s: str):
    nums = re.findall(r"-?\d+\.?\d*", s.replace(",", ""))
    return nums[-1] if nums else None


def _first_choice(s: str):
    m = re.search(r"\b([A-E])\b", s.upper())
    return m.group(1) if m else None

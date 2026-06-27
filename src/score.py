"""Score completions into 0/1 correctness, and extract a canonical answer LABEL.

Keep exact-match (deterministic) and LLM-judge (stochastic, separate analysis)
strictly apart -- this separation isolates sampling noise (H3) from judge noise (H2).

`extract_answer` returns the canonical answer label (number/letter) so the pipeline
can persist labels Y for the verifier-free aggregation oracle O^agg (Lemma aggfloor),
and `exact_match` is just `extract_answer(pred)==extract_answer(gold)`.
"""
from __future__ import annotations
import math
import re

_MATH = ("gsm8k", "math500", "mathbench")
_MC = ("mmlu_pro", "gpqa", "mmlu")
_CODE = ("mbpp", "humaneval", "livecodebench")


def extract_answer(s: str, task: str):
    """Canonical answer label for `s` under `task` (None if not found)."""
    s = str(s)
    if task in _MATH:
        return _norm_number(_math_number(s))
    if task in _MC:
        return _last_choice(s)
    return s.strip()                          # code / generic: raw text


def exact_match(prediction: str, gold: str, task: str) -> int:
    """Deterministic 0/1 score via canonical-label equality."""
    if task in _CODE:
        # TODO: run unit tests via a sandboxed executor; placeholder string match
        return int(prediction.strip() == str(gold).strip())
    p, g = extract_answer(prediction, task), extract_answer(gold, task)
    return int(p is not None and g is not None and p == g)


def judge_score(prediction: str, gold: str, judge_model: str) -> int:
    """LLM-as-judge for open-ended datasets. Stub -- wire the judge API."""
    raise NotImplementedError("Wire LLM-judge API; record judge_model + seed.")


def _last_number(s: str):
    nums = re.findall(r"-?\d+\.?\d*", s.replace(",", ""))
    return nums[-1] if nums else None


def _math_number(s: str):
    """Prefer the number after the LAST '####' marker the prompt asks for; else the
    last number anywhere (so a trailing distractor in free text is less likely to win)."""
    if "####" in s:
        n = _last_number(s.split("####")[-1])
        if n is not None:
            return n
    return _last_number(s)


def _norm_number(n):
    """Canonicalize a numeric string so '72', '72.0', '72.' all map to the same label.
    Guards huge digit strings that overflow float to inf/nan (e.g. a reasoning model
    emitting a 400-digit run) -> treat as 'no valid answer', never crash on round(inf)."""
    if n is None:
        return None
    try:
        f = float(n.rstrip("."))
    except (ValueError, OverflowError):
        return None
    if not math.isfinite(f):
        return None
    return str(int(round(f))) if abs(f - round(f)) < 1e-9 else repr(f)


def _last_choice(s: str):
    """Final multiple-choice letter: prefer the LAST 'answer/final ... <letter>' cue
    (NOT 'option/choice', which precede distractors like 'Option A is wrong'); else the
    LAST standalone A-E (so chain-of-thought that names a distractor first won't win)."""
    u = s.upper()
    cues = re.findall(r"(?:ANSWER|FINAL)\b[^A-E]{0,12}([A-E])\b", u)
    if cues:
        return cues[-1]
    ms = re.findall(r"\b([A-E])\b", u)
    return ms[-1] if ms else None

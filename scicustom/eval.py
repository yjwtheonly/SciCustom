"""Evaluate LLMs on a SciCustom-style MCQ benchmark.

Each item is a single-choice question whose answer is a single letter; we
prompt the candidate model, parse the predicted letter, and report accuracy.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, Sequence

from scicustom.llm import ChatModel
from scicustom.mcq import MCQItem


logger = logging.getLogger(__name__)


_EVAL_SYSTEM = (
    "You are taking a single-choice scientific exam.  Read the question "
    "carefully and respond with the single letter (A, B, C, D, or E) "
    "corresponding to the best answer.  Do not include any other text on "
    "the answer line."
)

_EVAL_USER = (
    "{query}\n\n"
    "Reply with the answer letter on the first line, then optionally a "
    "brief justification."
)


@dataclass
class PredictionRecord:
    query: str
    gold: str
    predicted: str | None
    raw_output: str
    correct: bool

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "gold": self.gold,
            "predicted": self.predicted,
            "raw_output": self.raw_output,
            "correct": self.correct,
        }


@dataclass
class EvalResult:
    model_name: str
    n_total: int
    n_correct: int
    predictions: list[PredictionRecord]

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_total if self.n_total else 0.0


_ANSWER_LETTER = re.compile(r"\b([A-Ea-e])\b")


def parse_letter(text: str) -> str | None:
    if not text:
        return None
    # Strip leading sentinels like "Answer: ".
    for prefix in ("answer:", "answer is", "final answer:"):
        idx = text.lower().find(prefix)
        if idx >= 0:
            text = text[idx + len(prefix):]
            break

    first_line = text.strip().splitlines()[0] if text.strip() else ""
    m = _ANSWER_LETTER.search(first_line)
    if m:
        return m.group(1).upper()
    # Last-ditch search over the entire output.
    m = _ANSWER_LETTER.search(text)
    return m.group(1).upper() if m else None


def evaluate_model(
    model: ChatModel,
    items: Sequence[MCQItem],
    *,
    max_workers: int = 8,
    temperature: float = 0.0,
    progress: bool = True,
) -> EvalResult:
    """Run a chat model over every MCQ and compute accuracy."""

    def _run(idx: int) -> PredictionRecord:
        item = items[idx]
        raw = model.chat(
            _EVAL_SYSTEM,
            _EVAL_USER.format(query=item.query),
            temperature=temperature,
            max_tokens=512,
        )
        pred = parse_letter(raw)
        return PredictionRecord(
            query=item.query,
            gold=item.answer,
            predicted=pred,
            raw_output=raw,
            correct=pred is not None and pred == item.answer,
        )

    records: list[PredictionRecord | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_run, i): i for i in range(len(items))}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                records[i] = fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.error("Eval call %d failed: %s", i, exc)
                records[i] = PredictionRecord(
                    query=items[i].query,
                    gold=items[i].answer,
                    predicted=None,
                    raw_output=str(exc),
                    correct=False,
                )
            done += 1
            if progress and done % max(1, len(items) // 20) == 0:
                logger.info("[eval %s] %d/%d done", model.name, done, len(items))

    final = [r for r in records if r is not None]
    n_correct = sum(1 for r in final if r.correct)
    return EvalResult(
        model_name=model.name,
        n_total=len(final),
        n_correct=n_correct,
        predictions=final,
    )


# ---------------------------------------------------------------------------
# Ranking consistency helpers (Section 3.1)
# ---------------------------------------------------------------------------

def spearman(ranks_a: Sequence[float], ranks_b: Sequence[float]) -> float:
    if len(ranks_a) != len(ranks_b) or len(ranks_a) < 2:
        return float("nan")
    a = _rank(ranks_a)
    b = _rank(ranks_b)
    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    num = sum((ai - mean_a) * (bi - mean_b) for ai, bi in zip(a, b))
    den_a = sum((ai - mean_a) ** 2 for ai in a)
    den_b = sum((bi - mean_b) ** 2 for bi in b)
    if den_a == 0 or den_b == 0:
        return float("nan")
    return num / (den_a * den_b) ** 0.5


def kendall_tau_b(ranks_a: Sequence[float], ranks_b: Sequence[float]) -> float:
    if len(ranks_a) != len(ranks_b) or len(ranks_a) < 2:
        return float("nan")
    a = list(ranks_a)
    b = list(ranks_b)
    concordant = discordant = ties_a = ties_b = 0
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif (da > 0) == (db > 0):
                concordant += 1
            else:
                discordant += 1
    denom_a = concordant + discordant + ties_a
    denom_b = concordant + discordant + ties_b
    if denom_a == 0 or denom_b == 0:
        return float("nan")
    return (concordant - discordant) / (denom_a * denom_b) ** 0.5


def _rank(xs: Sequence[float]) -> list[float]:
    indexed = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and xs[indexed[j + 1]] == xs[indexed[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def dump_results(results: Iterable[EvalResult], path: str) -> None:
    payload = []
    for r in results:
        payload.append(
            {
                "model": r.model_name,
                "n_total": r.n_total,
                "n_correct": r.n_correct,
                "accuracy": r.accuracy,
                "predictions": [p.to_dict() for p in r.predictions],
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %d model results to %s", len(payload), path)

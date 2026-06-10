"""MCQ transformation (Section 2.5, "Data-Grounded Benchmark Generation").

We hand the raw (query, answer) pair to an LLM and ask it to either preserve
the question (if already MCQ) or fabricate distractors (otherwise).  Only a
small wrapper around the prompt is needed at runtime; the heavy lifting is
done by the API call.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Sequence

from scicustom.llm import ChatModel
from scicustom.prompts import MCQ_SYSTEM, MCQ_USER
from scicustom.retrieval import DataInstance


logger = logging.getLogger(__name__)


_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")
_VALID_ANSWER = re.compile(r"^[A-E]$")


@dataclass
class MCQItem:
    query: str
    answer: str
    raw_source: str = ""
    source_idx: int | None = None
    meta: dict | None = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "raw_source": self.raw_source,
            "source_idx": self.source_idx,
            "meta": self.meta or {},
        }


def transform_one(model: ChatModel, instance: DataInstance, *, domain: str) -> MCQItem | None:
    """Convert a single (q, a) into a standardized MCQ via the LLM."""
    raw_content = f"Question: {instance.query}\nAnswer: {instance.answer}"
    response = model.chat(
        MCQ_SYSTEM.format(domain=domain),
        MCQ_USER.format(domain=domain, input_content=raw_content),
        temperature=0.0,
        max_tokens=1024,
    )

    parsed = _extract_json(response)
    if parsed is None:
        logger.warning("MCQ transform failed to parse JSON for query: %r", instance.query[:80])
        return None

    query = (parsed.get("query") or "").strip()
    answer = (parsed.get("answer") or "").strip().upper()
    if not query or not _VALID_ANSWER.match(answer):
        logger.warning("Discarding MCQ with invalid answer label %r", answer)
        return None

    return MCQItem(
        query=query,
        answer=answer,
        raw_source=raw_content,
        meta={"source": instance.source, "knowledge_units": instance.knowledge_units},
    )


def transform_batch(
    model: ChatModel,
    instances: Sequence[DataInstance],
    *,
    domain: str,
    max_workers: int = 8,
) -> list[MCQItem]:
    """Parallel MCQ transformation."""
    if not instances:
        return []

    results: list[MCQItem | None] = [None] * len(instances)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {
            pool.submit(transform_one, model, inst, domain=domain): i
            for i, inst in enumerate(instances)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                item = fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.error("MCQ transform raised for instance %d: %s", i, exc)
                item = None
            if item is not None:
                item.source_idx = i
            results[i] = item

    final = [r for r in results if r is not None]
    logger.info("MCQ transform: %d/%d successful", len(final), len(instances))
    return final


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    """Find the first JSON object in ``text``.

    LLMs occasionally pad the JSON with prose ("Sure, here's the result:\n
    {...}").  We accept that as long as the first balanced ``{...}`` block is
    valid JSON.
    """
    text = text.strip()
    # Markdown code fence trimmer.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    match = _JSON_BLOCK.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

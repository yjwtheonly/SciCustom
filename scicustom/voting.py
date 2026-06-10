"""Voting-based knowledge unit selection (Section 2.4).

Given a benchmark description, ask each judge LLM to rank the candidate tags,
average rank positions across judges, and return the top-K consensus units.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, Sequence

from scicustom.kb import KnowledgeBase, KnowledgeUnit, select_relevant_units
from scicustom.llm import ChatModel
from scicustom.prompts import VOTING_SYSTEM, VOTING_USER


logger = logging.getLogger(__name__)


@dataclass
class VotingResult:
    units: list[KnowledgeUnit]
    rank_table: dict[str, dict[str, int]]  # judge name -> {unit_id: rank}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_NUMBERED_LINE = re.compile(r"^\s*\d+[\.)]\s*")
_BULLET = re.compile(r"^\s*[\-\*•]\s*")


def parse_ranked_list(text: str) -> list[str]:
    """Pull out the ranked list of tag names from an LLM response.

    The voting prompt asks for "a single list of tags".  Models sometimes
    return a numbered list, sometimes bullets, sometimes a JSON array; we
    tolerate all three.
    """
    text = text.strip()

    # JSON array shortcut.
    if text.startswith("["):
        try:
            import json

            arr = json.loads(text)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if isinstance(x, (str, int, float))]
        except json.JSONDecodeError:
            pass

    lines = []
    for line in text.splitlines():
        line = _NUMBERED_LINE.sub("", line)
        line = _BULLET.sub("", line)
        line = line.strip().strip("`\"'")
        if not line:
            continue
        # Drop trailing ": ..." commentary if the model decorated the entry.
        line = line.split(":", 1)[0].strip()
        lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------

def rank_with_one_judge(
    model: ChatModel,
    domain: str,
    description: str,
    candidate_names: Sequence[str],
) -> list[str]:
    system = VOTING_SYSTEM.format(domain=domain)
    user = VOTING_USER.format(
        domain=domain,
        description=description,
        tag_list=", ".join(candidate_names),
    )
    raw = model.chat(system, user, temperature=0.0, max_tokens=2048)
    return parse_ranked_list(raw)


def vote_for_relevant_units(
    kb: KnowledgeBase,
    judges: Sequence[ChatModel],
    *,
    domain: str,
    description: str,
    candidate_names: Sequence[str] | None = None,
    top_k: int = 10,
    fuzzy_threshold: float = 85.0,
    max_workers: int = 4,
) -> VotingResult:
    """Top-K relevance ranking via multi-judge consensus.

    The candidate list mirrors what we used in the paper: by default we
    consider the entire KB, but you can pre-filter with frequency stats by
    passing ``candidate_names`` explicitly.
    """
    if not judges:
        raise ValueError("At least one judge model is required.")
    if candidate_names is None:
        candidate_names = kb.names

    logger.info("Running voting for domain=%r with %d judges over %d candidates",
                domain, len(judges), len(candidate_names))

    # Run judges concurrently - they are independent network calls.
    per_judge_ranks: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(rank_with_one_judge, judge, domain, description, candidate_names): judge.name
            for judge in judges
        }
        for fut in as_completed(futures):
            judge_name = futures[fut]
            try:
                per_judge_ranks[judge_name] = fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.error("Judge %s failed: %s; skipping its vote.", judge_name, exc)

    if not per_judge_ranks:
        raise RuntimeError("All judges failed - cannot aggregate votes.")

    # Map each judge's ranked names onto canonical knowledge units.  We keep
    # the rank a unit received from each judge for downstream debugging.
    n_judges = len(per_judge_ranks)
    rank_table: dict[str, dict[str, int]] = {j: {} for j in per_judge_ranks}
    aggregate_rank: dict[str, list[int]] = {}

    for judge_name, names in per_judge_ranks.items():
        for rank, name in enumerate(names):
            unit = kb.fuzzy_match(name, threshold=fuzzy_threshold)
            if unit is None:
                continue
            rank_table[judge_name].setdefault(unit.id, rank)
            aggregate_rank.setdefault(unit.id, []).append(rank)

    # Units that no judge ranked at all get pushed to the bottom.
    avg_rank: dict[str, float] = {}
    big = len(candidate_names) + 1
    for unit in kb:
        if unit.id in aggregate_rank:
            ranks = aggregate_rank[unit.id]
            # Missing judges -> their slot counts as the worst rank, matching
            # the "consensus rank" definition in Section 2.4.
            padded = ranks + [big] * (n_judges - len(ranks))
            avg_rank[unit.id] = sum(padded) / n_judges

    if not avg_rank:
        logger.warning("Voting produced no matches; falling back to the first judge's order.")
        first_judge_names = next(iter(per_judge_ranks.values()))
        return VotingResult(
            units=select_relevant_units(kb, first_judge_names, top_k=top_k, fuzzy_threshold=fuzzy_threshold),
            rank_table=rank_table,
        )

    ordered_ids = sorted(avg_rank, key=avg_rank.get)
    units: list[KnowledgeUnit] = []
    for uid in ordered_ids[:top_k]:
        u = next((x for x in kb if x.id == uid), None)
        if u is not None:
            units.append(u)

    logger.info("Voting picked top-%d units: %s", len(units), [u.name for u in units])
    return VotingResult(units=units, rank_table=rank_table)

"""End-to-end benchmark construction pipeline.

The online phase glue: parse a user requirement, vote on relevant knowledge
units, sort + cut the candidate corpus, pick a proxy subset, transform to
MCQs, and dump the resulting JSONL.

The offline phase (tagging the corpus once with :class:`SciTagger`) is
expected to run separately - the resulting tagged corpus is fed to
:func:`build_benchmark` as ``corpus``.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Sequence

from scicustom.kb import KnowledgeBase, load_knowledge_units
from scicustom.llm import ChatModel
from scicustom.mcq import MCQItem, transform_batch
from scicustom.retrieval import (
    DataInstance,
    binary_search_cutoff,
    select_proxy_subset,
    sort_candidates,
)
from scicustom.voting import VotingResult, vote_for_relevant_units


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkArtifacts:
    """Bundle returned by :func:`build_benchmark`.

    We keep the intermediate artifacts around so that downstream debugging
    and the figures in Section 3.4 can be regenerated from a single run.
    """

    requirement: str
    voting: VotingResult
    sorted_candidates: list[DataInstance]
    cutoff: int
    retrieved: list[DataInstance]
    proxy_subset: list[DataInstance]
    mcqs: list[MCQItem]
    config: dict = field(default_factory=dict)

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "voting.json"), "w") as f:
            json.dump(
                {
                    "units": [u.name for u in self.voting.units],
                    "rank_table": self.voting.rank_table,
                },
                f,
                indent=2,
            )
        _dump_jsonl([_di_to_dict(d) for d in self.retrieved], os.path.join(directory, "retrieved.jsonl"))
        _dump_jsonl([_di_to_dict(d) for d in self.proxy_subset], os.path.join(directory, "proxy.jsonl"))
        _dump_jsonl([m.to_dict() for m in self.mcqs], os.path.join(directory, "benchmark.jsonl"))
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
        logger.info("Saved benchmark artifacts to %s", directory)


def _di_to_dict(d: DataInstance) -> dict:
    return {
        "query": d.query,
        "answer": d.answer,
        "knowledge_units": list(d.knowledge_units),
        "source": d.source,
        "meta": d.meta,
    }


def _dump_jsonl(rows: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_benchmark(
    *,
    requirement: str,
    description: str,
    domain: str,
    corpus: Sequence[DataInstance],
    voting_judges: Sequence[ChatModel],
    relevance_judges: Sequence[ChatModel],
    mcq_writer: ChatModel,
    knowledge_base: KnowledgeBase | None = None,
    k1: int = 10,
    k2: int = 100,
    fuzzy_threshold: float = 85.0,
    proxy_samples: int = 100,
    encoder_name: str = "roberta-base",
    out_dir: str | None = None,
) -> BenchmarkArtifacts:
    """Build a SciCustom benchmark for a single requirement.

    Parameters mirror Appendix C: ``k1`` is the voting top-K, ``k2`` is the
    proxy subset target size, the fuzzy threshold is 85 (rapidfuzz Indel
    similarity), and the proxy stage samples 100 candidate subsets.
    """
    kb = knowledge_base or load_knowledge_units()
    logger.info("Pipeline start: requirement=%r domain=%s", requirement, domain)

    # 1. Voting.
    voting = vote_for_relevant_units(
        kb,
        voting_judges,
        domain=domain,
        description=description,
        top_k=k1,
        fuzzy_threshold=fuzzy_threshold,
    )

    # 2. Sort the corpus by overlap + rank with the voted units.
    rank_of_unit = {u.id: i for i, u in enumerate(voting.units)}
    sorted_corpus = sort_candidates(corpus, voting.units, rank_of_unit=rank_of_unit)
    # The binary search only makes sense when at least one unit overlaps.
    matched_prefix = [
        d for d in sorted_corpus
        if any(t in {u.name for u in voting.units} for t in d.knowledge_units)
    ]
    logger.info("Sorted corpus: |D|=%d, with-overlap=%d", len(sorted_corpus), len(matched_prefix))

    # 3. Binary-search cutoff for relevance.
    cutoff = binary_search_cutoff(
        matched_prefix,
        relevance_judges,
        requirement=requirement,
        description=description,
    )
    retrieved = matched_prefix[:cutoff]

    if not retrieved:
        logger.warning("Cutoff produced an empty set; falling back to top-100 overlap matches.")
        retrieved = matched_prefix[:100]

    # 4. Proxy subset.
    proxy = select_proxy_subset(
        retrieved,
        k2=k2,
        n_samples=proxy_samples,
        encoder_name=encoder_name,
    )

    # 5. MCQ transformation.
    logger.info("Transforming %d proxy items into MCQs via %s", len(proxy), mcq_writer.name)
    mcqs = transform_batch(mcq_writer, proxy, domain=domain)

    artifacts = BenchmarkArtifacts(
        requirement=requirement,
        voting=voting,
        sorted_candidates=sorted_corpus,
        cutoff=cutoff,
        retrieved=retrieved,
        proxy_subset=proxy,
        mcqs=mcqs,
        config={
            "requirement": requirement,
            "description": description,
            "domain": domain,
            "k1": k1,
            "k2": k2,
            "fuzzy_threshold": fuzzy_threshold,
            "proxy_samples": proxy_samples,
            "encoder_name": encoder_name,
            "voting_judges": [j.name for j in voting_judges],
            "relevance_judges": [j.name for j in relevance_judges],
            "mcq_writer": mcq_writer.name,
        },
    )

    if out_dir:
        artifacts.save(out_dir)

    logger.info("Pipeline done.  %d MCQs written.", len(mcqs))
    return artifacts


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_tagged_corpus(path: str) -> list[DataInstance]:
    """Read a JSONL file containing tagged (q, a) records produced by the
    offline indexing phase (see ``scripts/tag_queries.py``)."""

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    instances: list[DataInstance] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.error("Bad JSON on line %d of %s: %s", ln, path, exc)
                continue
            instances.append(
                DataInstance(
                    query=row.get("query") or row.get("question") or "",
                    answer=row.get("answer") or "",
                    knowledge_units=list(row.get("knowledge_units") or row.get("tags") or []),
                    source=row.get("source", ""),
                    meta=row.get("meta", {}) or {},
                )
            )
    logger.info("Loaded %d tagged instances from %s", len(instances), path)
    return instances

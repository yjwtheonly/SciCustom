"""Hierarchical benchmark retrieval (Section 2.5).

This module implements two ideas:

* Algorithm 2: binary search over a sorted candidate list to find the
  relevance cutoff with a small number of LLM-judge oracle calls.
* SubLIME-style proxy subset selection: pick a representative subset of size
  K2 from the retrieved data by clustering RoBERTa embeddings and selecting
  cluster heads under a Wasserstein-distance objective on (hardness, quality).
"""
from __future__ import annotations

import logging
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from scicustom.kb import KnowledgeBase, KnowledgeUnit
from scicustom.llm import ChatModel


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DataInstance:
    """A scientific QA pair plus its tagger-assigned knowledge units."""

    query: str
    answer: str
    knowledge_units: list[str] = field(default_factory=list)
    # Optional extra fields kept around for downstream evaluation/debugging.
    source: str = ""
    meta: dict = field(default_factory=dict)

    def overlap(self, required: set[str]) -> set[str]:
        return required.intersection(self.knowledge_units)


# ---------------------------------------------------------------------------
# Sorting (Section 2.5, "ordering rule")
# ---------------------------------------------------------------------------

def sort_candidates(
    candidates: Sequence[DataInstance],
    required_units: Sequence[KnowledgeUnit],
    rank_of_unit: dict[str, float] | None = None,
) -> list[DataInstance]:
    """Order the candidate set by (|T_r,q|, avg rank).

    Implements the two-key sort from Section 2.5:
    1. Higher knowledge-unit overlap with the requirement wins.
    2. Ties are broken by lower average voting rank of the matched units
       (i.e., more relevant on average).
    """
    required_ids = {u.id for u in required_units}
    required_names = {u.name for u in required_units}
    name_to_id = {u.name: u.id for u in required_units}
    name_to_id.update({n: n for n in required_names})

    if rank_of_unit is None:
        rank_of_unit = {u.id: i for i, u in enumerate(required_units)}

    def sort_key(d: DataInstance):
        matched = [name_to_id.get(t, t) for t in d.knowledge_units if t in name_to_id or t in required_ids]
        size = len(matched)
        if size == 0:
            avg_rank = math.inf
        else:
            ranks = [rank_of_unit.get(m, len(required_units)) for m in matched]
            avg_rank = sum(ranks) / size
        return (-size, avg_rank)

    return sorted(candidates, key=sort_key)


# ---------------------------------------------------------------------------
# Algorithm 2: Binary search relevance cutoff
# ---------------------------------------------------------------------------

def binary_search_cutoff(
    sorted_list: Sequence[DataInstance],
    judges: Sequence[ChatModel],
    *,
    requirement: str,
    description: str = "",
    max_workers: int = 4,
    verbose: bool = True,
) -> int:
    """Find the cutoff position via binary search (Algorithm 2).

    Returns the cutoff index ``c`` such that ``sorted_list[: c]`` are the data
    instances retained for the benchmark.  ``c`` is zero-indexed and counts
    from the start of the sorted list.
    """
    low, high = 0, len(sorted_list) - 1
    cutoff = 0
    n_calls = 0

    if len(sorted_list) == 0:
        return 0

    while low <= high:
        mid = low + (high - low) // 2
        instance = sorted_list[mid]
        is_relevant = _majority_relevance_vote(
            judges, requirement, description, instance, max_workers=max_workers
        )
        n_calls += 1
        if verbose:
            logger.info("[bsearch] step=%d range=[%d, %d] mid=%d -> %s",
                        n_calls, low, high, mid, "relevant" if is_relevant else "irrelevant")
        if is_relevant:
            cutoff = mid + 1
            low = mid + 1
        else:
            high = mid - 1

    logger.info("Binary search converged: cutoff=%d / N=%d (%d oracle calls)",
                cutoff, len(sorted_list), n_calls)
    return cutoff


_RELEVANCE_SYSTEM = (
    "You judge whether a single (question, answer) data point is RELEVANT "
    "for evaluating an LLM on a given scientific requirement.  Answer with "
    "`YES` or `NO` followed by a one-sentence justification."
)

_RELEVANCE_USER = (
    "Scientific requirement: {requirement}\n"
    "Description: {description}\n\n"
    "Question: {question}\n"
    "Answer: {answer}\n\n"
    "Is this data point relevant to the requirement?  Reply with `YES` or "
    "`NO` only on the first line."
)


def _ask_one_judge(model: ChatModel, requirement: str, description: str, instance: DataInstance) -> bool:
    raw = model.chat(
        _RELEVANCE_SYSTEM,
        _RELEVANCE_USER.format(
            requirement=requirement,
            description=description or requirement,
            question=instance.query,
            answer=instance.answer,
        ),
        temperature=0.0,
        max_tokens=128,
    )
    first = raw.strip().splitlines()[0].strip().lower() if raw else ""
    return first.startswith("yes")


def _majority_relevance_vote(
    judges: Sequence[ChatModel],
    requirement: str,
    description: str,
    instance: DataInstance,
    max_workers: int = 4,
) -> bool:
    if not judges:
        return True

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        votes = list(pool.map(lambda m: _ask_one_judge(m, requirement, description, instance), judges))

    yes_votes = sum(1 for v in votes if v)
    return yes_votes > len(judges) // 2


# ---------------------------------------------------------------------------
# Proxy subset selection (Section 2.5, "Efficient Proxy Subset Selection")
# ---------------------------------------------------------------------------

def perplexity_hardness(
    instance: DataInstance,
    *,
    score_fn: Callable[[str, str], float] | None = None,
) -> float:
    """Wrapper around a configurable PPL backend.

    The paper computes ``H(d) = exp(- 1/|a_d| * sum log P(t_i | q_d, t_<i))``
    using an open-weights LLM.  We do not ship a default scorer (it would
    require shipping yet another model) - callers should pass one in via
    ``score_fn``.  When omitted we fall back to a length-based heuristic.
    """
    if score_fn is not None:
        return score_fn(instance.query, instance.answer)

    # Fallback - normalized inverse length, just to keep the pipeline running.
    n = max(len(instance.answer.split()), 1)
    return math.exp(-1.0 / n)


def flesch_reading_ease(text: str) -> float:
    """Compute the Flesch Reading Ease score.

    We avoid pulling in textstat to keep dependencies light - the formula is
    well-known.  Higher scores indicate easier-to-read text.
    """
    if not text:
        return 0.0

    sentences = max(text.count(".") + text.count("!") + text.count("?"), 1)
    words = text.split()
    n_words = max(len(words), 1)
    n_syllables = sum(_count_syllables(w) for w in words)

    return 206.835 - 1.015 * (n_words / sentences) - 84.6 * (n_syllables / n_words)


_VOWELS = set("aeiouy")


def _count_syllables(word: str) -> int:
    word = word.lower().strip(".,;:!?\"'()[]")
    if not word:
        return 0
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in _VOWELS
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _embed_corpus(
    instances: Sequence[DataInstance],
    *,
    encoder_name: str = "roberta-base",
    batch_size: int = 32,
    device: str | None = None,
) -> np.ndarray:
    """Encode QA pairs with a RoBERTa model (sentence-level CLS pooling)."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Proxy subset selection requires transformers + torch.  "
            "Install with `pip install transformers torch`."
        ) from exc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    model = AutoModel.from_pretrained(encoder_name).to(device)
    model.eval()

    texts = [f"{d.query}\n{d.answer}" for d in instances]
    embeds = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = tokenizer(chunk, padding=True, truncation=True, return_tensors="pt", max_length=256)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()
            embeds.append(cls)
    return np.concatenate(embeds, axis=0)


def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Cumulative 1-D Wasserstein distance (a.k.a. Earth Mover's distance)."""
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    # Resample to common length so we can compare cumulative distributions.
    n = max(len(a_sorted), len(b_sorted))
    a_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(a_sorted)), a_sorted)
    b_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(b_sorted)), b_sorted)
    return float(np.mean(np.abs(a_interp - b_interp)))


def select_proxy_subset(
    candidates: Sequence[DataInstance],
    *,
    k2: int = 100,
    n_samples: int = 100,
    encoder_name: str = "roberta-base",
    hardness_fn: Callable[[DataInstance], float] | None = None,
    seed: int = 0,
) -> list[DataInstance]:
    """Pick a representative subset of size K2 minimizing Wasserstein gap.

    Implements the SubLIME-flavored procedure from Section 2.5 and Appendix
    A.  We cluster RoBERTa embeddings with k-means and sample one
    representative from each cluster, then evaluate the cumulative
    Wasserstein distance against the full population on the (hardness,
    quality) score distributions.  The best of ``n_samples`` candidate
    subsets is returned.
    """
    if len(candidates) == 0:
        return []
    if len(candidates) <= k2:
        logger.info("Proxy subset selection: |D_r|=%d <= K2=%d, returning all candidates.",
                    len(candidates), k2)
        return list(candidates)

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Scoring
    hardness_fn = hardness_fn or perplexity_hardness
    hardness = np.array([hardness_fn(d) for d in candidates])
    quality = np.array([flesch_reading_ease(d.answer) for d in candidates])

    # Embedding + clustering
    embeds = _embed_corpus(candidates, encoder_name=encoder_name)
    try:
        from sklearn.cluster import KMeans  # type: ignore
    except ImportError as exc:
        raise ImportError("Please `pip install scikit-learn` for clustering.") from exc

    n_clusters = min(k2, len(candidates))
    logger.info("Clustering %d candidates into %d groups for proxy subset selection",
                len(candidates), n_clusters)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=4)
    labels = km.fit_predict(embeds)

    # Build per-cluster index lists once and re-sample from them.
    by_cluster: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels):
        by_cluster.setdefault(int(lab), []).append(idx)

    best_subset: list[int] | None = None
    best_score = math.inf
    for _ in range(n_samples):
        picked: list[int] = []
        for cluster_id, idxs in by_cluster.items():
            picked.append(rng.choice(idxs))
        if len(picked) > k2:
            picked = rng.sample(picked, k2)
        elif len(picked) < k2:
            # Fill up with random extras from the global pool.
            extras = list(set(range(len(candidates))) - set(picked))
            picked.extend(rng.sample(extras, k2 - len(picked)))

        sub_hardness = hardness[picked]
        sub_quality = quality[picked]
        score = _wasserstein_1d(hardness, sub_hardness) + _wasserstein_1d(quality, sub_quality)
        if score < best_score:
            best_score = score
            best_subset = picked

    logger.info("Selected proxy subset of size %d with Wasserstein gap %.4f", k2, best_score)
    assert best_subset is not None
    return [candidates[i] for i in best_subset]

"""Ontology-grounded knowledge unit selection (Section 2.2).

Implements Algorithm 1 from the paper: traverse the input DAGs depth-first,
classify each visited node into ``coarse / moderate / fine`` with an LLM, and
keep the moderate-granularity terms.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from scicustom.llm import ChatModel
from scicustom.prompts import GRANULARITY_SYSTEM, GRANULARITY_USER


logger = logging.getLogger(__name__)


MIN_DESCENDANTS = 10  # Algorithm 1: "if |Desc(v)| < 10 then Backtrack"


@dataclass
class OntologyNode:
    id: str
    name: str
    children: list["OntologyNode"] = field(default_factory=list)

    def descendants(self) -> list["OntologyNode"]:
        out: list[OntologyNode] = []
        stack = list(self.children)
        seen: set[str] = set()
        while stack:
            node = stack.pop()
            if node.id in seen:
                continue
            seen.add(node.id)
            out.append(node)
            stack.extend(node.children)
        return out


@dataclass
class GranularityResult:
    label: str  # "moderate" | "coarse" | "fine"
    explanation: str = ""
    raw: str = ""


_GRANULARITY_RE = re.compile(r"\(?\s*(moderate|too\s*coarse|too\s*fine|coarse|fine)\s*\)?", re.IGNORECASE)


def parse_granularity(text: str) -> GranularityResult:
    """Parse the output of the granularity-classification prompt."""
    raw = text.strip()
    m = _GRANULARITY_RE.search(raw)
    if not m:
        logger.warning("Could not parse granularity label from: %r", raw[:120])
        return GranularityResult(label="unknown", raw=raw)
    label = m.group(1).lower().replace(" ", "")
    if label == "moderate":
        norm = "moderate"
    elif label.startswith("too"):
        norm = "coarse" if "coarse" in label else "fine"
    elif label == "coarse":
        norm = "coarse"
    elif label == "fine":
        norm = "fine"
    else:
        norm = "unknown"

    expl = ""
    if "Explanation" in raw:
        expl = raw.split("Explanation", 1)[1].lstrip(":; ").strip()
    return GranularityResult(label=norm, explanation=expl, raw=raw)


def classify_node(model: ChatModel, term: str) -> GranularityResult:
    """Run the LLM granularity classifier on a single term."""
    user = GRANULARITY_USER.format(term=term)
    out = model.chat(GRANULARITY_SYSTEM, user, temperature=0.0, max_tokens=256)
    return parse_granularity(out)


def select_knowledge_units(
    roots: Iterable[OntologyNode],
    model: ChatModel,
    *,
    min_descendants: int = MIN_DESCENDANTS,
) -> list[OntologyNode]:
    """Run Algorithm 1 end-to-end on a forest of ontology DAGs.

    Returns the set of moderate-granularity nodes that will become the
    SciCustom knowledge units.  The traversal is depth-first; "coarse"
    nodes recurse into their children, "moderate" nodes are kept (and we
    do NOT recurse further), and "fine" nodes prune their subtree.
    """
    kept: list[OntologyNode] = []
    seen: set[str] = set()
    n_calls = defaultdict(int)

    def dfs(node: OntologyNode, depth: int = 0) -> None:
        if node.id in seen:
            return
        seen.add(node.id)

        descs = node.descendants()
        if len(descs) < min_descendants:
            # Too small to be a useful knowledge unit; back off.
            return

        n_calls["llm"] += 1
        result = classify_node(model, node.name)
        logger.debug("[ontology] depth=%d term=%r -> %s", depth, node.name, result.label)

        if result.label == "coarse":
            for child in node.children:
                dfs(child, depth + 1)
        elif result.label == "moderate":
            kept.append(node)
            # Do not recurse - keep granularity uniform.
        else:
            # "fine" or "unknown" -> prune branch.
            return

    for root in roots:
        dfs(root)

    logger.info("Ontology selection complete: %d knowledge units (%d LLM calls)",
                len(kept), n_calls["llm"])
    return kept

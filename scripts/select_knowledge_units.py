#!/usr/bin/env python
"""Run Algorithm 1 (granularity-driven DFS) over a forest of ontology DAGs.

Each DAG is read from a JSON file with this minimal schema::

    {
      "root": {"id": "chemistry", "name": "Chemistry"},
      "edges": [
        {"parent": "chemistry", "child": "organic_chemistry"},
        ...
      ],
      "labels": {"organic_chemistry": "Organic Chemistry", ...}
    }

The script visits each DAG, asks the chosen LLM whether each node is
``coarse``, ``moderate`` or ``fine``, and writes the kept "moderate" nodes to
the output JSON.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scicustom.llm import get_chat_model
from scicustom.ontology import OntologyNode, select_knowledge_units
from scicustom.utils import configure_logging


logger = logging.getLogger("scicustom.ontology")


def _build_nodes(payload: dict) -> OntologyNode:
    labels = payload.get("labels", {})
    nodes: dict[str, OntologyNode] = {}

    def get(nid: str) -> OntologyNode:
        if nid not in nodes:
            nodes[nid] = OntologyNode(id=nid, name=labels.get(nid, nid))
        return nodes[nid]

    for edge in payload.get("edges", []):
        parent = get(edge["parent"])
        child = get(edge["child"])
        parent.children.append(child)

    root_id = payload["root"]["id"]
    if root_id not in nodes:
        nodes[root_id] = OntologyNode(id=root_id, name=payload["root"].get("name", root_id))
    return nodes[root_id]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dag", action="append", required=True, help="Path to a DAG JSON file (repeat for forest)")
    p.add_argument("--model", required=True, help="LLM id used as the granularity classifier (e.g. gpt-5)")
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument("--min-descendants", type=int, default=10)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    configure_logging(args.log_level)
    model = get_chat_model(args.model)

    roots = []
    for path in args.dag:
        with open(path) as f:
            payload = json.load(f)
        roots.append(_build_nodes(payload))
        logger.info("Loaded DAG from %s", path)

    kept = select_knowledge_units(roots, model, min_descendants=args.min_descendants)

    payload = {
        "knowledge_units": [
            {"id": n.id, "name": n.name, "domain": "auto"} for n in kept
        ],
        "n_units": len(kept),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[ontology] kept {len(kept)} knowledge units -> {args.out}")


if __name__ == "__main__":
    main()

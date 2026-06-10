#!/usr/bin/env python
"""Tag a corpus of scientific queries with the SciCustom tagger.

Reads either a plain-text file (one query per line) or a JSONL file with
``query`` / ``answer`` keys and writes a JSONL file whose rows include the
predicted ``knowledge_units``.

Example:
    python scripts/tag_queries.py \\
        --input data/sciriff.jsonl \\
        --output runs/sciriff.tagged.jsonl \\
        --batch-size 64
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Allow running directly without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scicustom.tagger import SciTagger, DEFAULT_MODEL_ID
from scicustom.utils import configure_logging, load_jsonl, write_jsonl


logger = logging.getLogger("scicustom.tag")


def _iter_rows(path: str) -> list[dict]:
    if path.endswith(".jsonl") or path.endswith(".json"):
        rows = load_jsonl(path) if path.endswith(".jsonl") else json.load(open(path))
        if not isinstance(rows, list):
            raise ValueError(f"Expected a list of records in {path}")
        return rows
    # Plain text: one query per line.
    with open(path, "r", encoding="utf-8") as f:
        return [{"query": ln.strip()} for ln in f if ln.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Input file (.jsonl, .json, or .txt)")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument("--model", default=DEFAULT_MODEL_ID, help="HF model id of the tagger")
    p.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--max-tags", type=int, default=8)
    p.add_argument("--fuzzy-threshold", type=float, default=85.0)
    p.add_argument("--knowledge-units", default=None, help="Path to a custom knowledge_units.json (optional)")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--limit", type=int, default=None, help="Cap on the number of input queries (debug)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    configure_logging(args.log_level)

    rows = _iter_rows(args.input)
    if args.limit:
        rows = rows[: args.limit]
    queries = [str(r.get("query") or r.get("question") or "").strip() for r in rows]
    queries = [q for q in queries if q]
    logger.info("Read %d queries from %s", len(queries), args.input)

    backend_kwargs = {}
    if args.backend == "vllm":
        backend_kwargs = {
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        }

    tagger = SciTagger.from_pretrained(
        args.model,
        backend=args.backend,
        knowledge_base=args.knowledge_units,
        fuzzy_threshold=args.fuzzy_threshold,
        max_tags=args.max_tags,
        **backend_kwargs,
    )

    output_rows: list[dict] = []
    for start in range(0, len(queries), args.batch_size):
        batch = queries[start : start + args.batch_size]
        results = tagger.tag(batch, max_tokens=args.max_tokens)
        # In batch mode the tagger returns TagResult objects.
        for src_row, qstr, res in zip(rows[start : start + args.batch_size], batch, results):
            out = dict(src_row)
            out["query"] = qstr
            out["knowledge_units"] = res.names
            out["tagger_raw"] = res.raw_output
            output_rows.append(out)
        logger.info("Tagged %d / %d (last batch tags=%s)", min(start + args.batch_size, len(queries)),
                    len(queries), results[-1].names if results else [])

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    write_jsonl(output_rows, args.output)
    print(f"[tag] wrote {len(output_rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()

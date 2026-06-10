#!/usr/bin/env python
"""Build a SciCustom benchmark from a tagged corpus and a config file.

Example:
    python scripts/build_benchmark.py \\
        --config configs/chemistry/organic.yaml \\
        --corpus runs/sciriff.tagged.jsonl \\
        --out-dir runs/organic_chemistry
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scicustom.llm import get_chat_model
from scicustom.pipeline import build_benchmark, load_tagged_corpus
from scicustom.utils import configure_logging, load_yaml


logger = logging.getLogger("scicustom.build")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="YAML config defining the requirement and judges")
    p.add_argument("--corpus", required=True, help="JSONL produced by `scripts/tag_queries.py`")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--k1", type=int, default=None, help="Override voting top-K")
    p.add_argument("--k2", type=int, default=None, help="Override proxy subset size")
    p.add_argument("--encoder", default=None, help="HF id of the embedding model for subset selection")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    configure_logging(args.log_level)
    cfg = load_yaml(args.config)
    logger.info("Loaded config: %s", args.config)

    corpus = load_tagged_corpus(args.corpus)

    voting_judges = [get_chat_model(spec) for spec in cfg["voting_judges"]]
    relevance_judges = [get_chat_model(spec) for spec in cfg.get("relevance_judges", cfg["voting_judges"])]
    mcq_writer = get_chat_model(cfg["mcq_writer"])

    artifacts = build_benchmark(
        requirement=cfg["requirement"],
        description=cfg.get("description", ""),
        domain=cfg.get("domain", cfg["requirement"]),
        corpus=corpus,
        voting_judges=voting_judges,
        relevance_judges=relevance_judges,
        mcq_writer=mcq_writer,
        k1=args.k1 or cfg.get("k1", 10),
        k2=args.k2 or cfg.get("k2", 100),
        fuzzy_threshold=cfg.get("fuzzy_threshold", 85.0),
        proxy_samples=cfg.get("proxy_samples", 100),
        encoder_name=args.encoder or cfg.get("encoder_name", "roberta-base"),
        out_dir=args.out_dir,
    )

    print(f"[build] benchmark written to {args.out_dir} ({len(artifacts.mcqs)} MCQs)")


if __name__ == "__main__":
    main()

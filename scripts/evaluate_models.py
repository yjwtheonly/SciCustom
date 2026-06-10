#!/usr/bin/env python
"""Run a list of LLMs on a SciCustom MCQ benchmark and dump accuracies.

Example:
    python scripts/evaluate_models.py \\
        --benchmark runs/organic_chemistry/benchmark.jsonl \\
        --models configs/eval_models.yaml \\
        --out runs/organic_chemistry/eval.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scicustom.eval import dump_results, evaluate_model
from scicustom.llm import get_chat_model
from scicustom.mcq import MCQItem
from scicustom.utils import configure_logging, load_jsonl, load_yaml


logger = logging.getLogger("scicustom.evaluate")


def _load_benchmark(path: str) -> list[MCQItem]:
    rows = load_jsonl(path)
    items: list[MCQItem] = []
    for r in rows:
        items.append(
            MCQItem(
                query=r["query"],
                answer=r["answer"],
                raw_source=r.get("raw_source", ""),
                meta=r.get("meta"),
            )
        )
    return items


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--models", required=True, help="YAML file with the model list")
    p.add_argument("--out", required=True)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    configure_logging(args.log_level)

    items = _load_benchmark(args.benchmark)
    logger.info("Loaded %d MCQs from %s", len(items), args.benchmark)

    spec_cfg = load_yaml(args.models)
    specs = spec_cfg["models"] if "models" in spec_cfg else spec_cfg
    if isinstance(specs, dict):
        # A dict-of-name -> spec; normalize to list.
        specs = list(specs.values())

    summary: dict[str, float] = {}
    all_results = []
    for spec in specs:
        try:
            chat = get_chat_model(spec)
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not initialize model %r: %s", spec, exc)
            continue
        print(f"[evaluate] {chat.name}: starting", flush=True)
        result = evaluate_model(
            chat,
            items,
            max_workers=args.max_workers,
            temperature=args.temperature,
        )
        summary[chat.name] = result.accuracy
        all_results.append(result)
        print(f"[evaluate] {chat.name}: accuracy={result.accuracy:.3f} ({result.n_correct}/{result.n_total})",
              flush=True)

    dump_results(all_results, args.out)
    summary_path = Path(args.out).with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[evaluate] summary -> {summary_path}")


if __name__ == "__main__":
    main()

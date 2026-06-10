#!/usr/bin/env python
"""Evaluate the tagger against gold annotations using fuzzy label matching.

The gold file is JSONL with rows of ``{"query": ..., "knowledge_units": [...]}``.
Predictions are matched to gold tags by rapidfuzz Indel similarity with the
threshold reported in Section 3.3 (default 85).

We report Macro / Micro F1 - matching Table 4 of Appendix D in spirit
(numbers will differ depending on the test split used).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover
    fuzz = None

from scicustom.kb import load_knowledge_units
from scicustom.tagger import SciTagger, DEFAULT_MODEL_ID
from scicustom.utils import configure_logging, load_jsonl


logger = logging.getLogger("scicustom.eval_tagger")


def _fuzzy_in(name: str, gold_set: set[str], threshold: float) -> bool:
    if name in gold_set:
        return True
    if fuzz is None:
        return False
    for g in gold_set:
        if fuzz.ratio(name, g) >= threshold:
            return True
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold", required=True, help="JSONL gold file")
    p.add_argument("--model", default=DEFAULT_MODEL_ID)
    p.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--knowledge-units", default=None)
    p.add_argument("--threshold", type=float, default=85.0,
                   help="Indel similarity threshold (paper uses 85)")
    p.add_argument("--out", default=None, help="Optional predictions JSONL")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    configure_logging(args.log_level)

    gold_rows = load_jsonl(args.gold)
    queries = [r["query"] for r in gold_rows]
    gold_sets: list[set[str]] = [set(r["knowledge_units"]) for r in gold_rows]

    kb = load_knowledge_units(args.knowledge_units)
    tagger = SciTagger.from_pretrained(
        args.model,
        backend=args.backend,
        knowledge_base=kb,
    )

    pred_sets: list[set[str]] = []
    raw_preds: list[list[str]] = []
    for start in range(0, len(queries), args.batch_size):
        batch = queries[start : start + args.batch_size]
        results = tagger.tag(batch)
        for res in results:
            names = res.names
            pred_sets.append(set(names))
            raw_preds.append(names)
        logger.info("[eval_tagger] %d/%d done", min(start + args.batch_size, len(queries)), len(queries))

    # Per-tag tp/fp/fn for macro F1.
    tag_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    micro = {"tp": 0, "fp": 0, "fn": 0}
    for pred, gold in zip(pred_sets, gold_sets):
        for g in gold:
            hit = _fuzzy_in(g, pred, args.threshold)
            if hit:
                tag_counts[g]["tp"] += 1
                micro["tp"] += 1
            else:
                tag_counts[g]["fn"] += 1
                micro["fn"] += 1
        for ppred in pred:
            if not _fuzzy_in(ppred, gold, args.threshold):
                tag_counts[ppred]["fp"] += 1
                micro["fp"] += 1

    def f1(stats):
        prec = stats["tp"] / max(stats["tp"] + stats["fp"], 1)
        rec = stats["tp"] / max(stats["tp"] + stats["fn"], 1)
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)

    macro = sum(f1(v) for v in tag_counts.values()) / max(len(tag_counts), 1)
    micro_f1 = f1(micro)

    print(f"[eval_tagger] Macro F1 = {macro*100:.2f}")
    print(f"[eval_tagger] Micro F1 = {micro_f1*100:.2f}")
    print(f"[eval_tagger] {len(tag_counts)} distinct tags observed across pred+gold")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for q, gold, pred in zip(queries, gold_sets, raw_preds):
                f.write(
                    json.dumps(
                        {
                            "query": q,
                            "gold": sorted(gold),
                            "pred": pred,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"[eval_tagger] wrote predictions to {args.out}")


if __name__ == "__main__":
    main()

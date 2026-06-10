"""Miscellaneous helpers."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Iterable

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is in requirements but may be absent
    yaml = None  # type: ignore[assignment]


def configure_logging(level: str = "INFO") -> None:
    """Set up a sensible default logging config for the CLI scripts.

    We intentionally print to stdout so the logs interleave with tqdm bars
    and shell tee pipelines behave reasonably.
    """
    if logging.getLogger().handlers:
        # Already configured (e.g., by an outer framework).  Just bump levels.
        logging.getLogger().setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def load_yaml(path: str) -> dict:
    if yaml is None:
        raise ImportError("PyYAML is required to load YAML configs.  `pip install pyyaml`")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@contextmanager
def timed(msg: str):
    """A tiny context manager for the kinds of "how long did that take?"
    spot checks we sprinkle through the scripts."""
    t0 = time.time()
    print(f"[timed] start: {msg}", flush=True)
    try:
        yield
    finally:
        dt = time.time() - t0
        print(f"[timed] done : {msg}  ({dt:.2f}s)", flush=True)


def safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

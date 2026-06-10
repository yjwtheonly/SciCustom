"""Knowledge base helpers.

The 642 ontology-grounded knowledge units are shipped as a small JSON file
under ``assets/knowledge_units.json``.  This module loads them and exposes a
``KnowledgeBase`` object with a couple of conveniences used by the tagger.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from importlib import resources
from typing import Iterable, Sequence

try:  # optional dependency, only needed for fuzzy matching
    from rapidfuzz import fuzz, process
except ImportError:  # pragma: no cover - degrade gracefully when not installed
    fuzz = None
    process = None


logger = logging.getLogger(__name__)


@dataclass
class KnowledgeUnit:
    id: str
    name: str
    domain: str = "unknown"
    aliases: list[str] = field(default_factory=list)

    @property
    def canonical(self) -> str:
        return self.name


# A cheap fingerprint used inside the package for membership / dedup.
def _norm(s: str) -> str:
    return " ".join(s.lower().strip().split())


@dataclass
class KnowledgeBase:
    units: list[KnowledgeUnit]
    version: str = "0.0.0"

    # filled lazily by __post_init__
    _name_to_unit: dict[str, KnowledgeUnit] = field(init=False, repr=False)
    _all_names: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._name_to_unit = {}
        for u in self.units:
            self._name_to_unit[_norm(u.name)] = u
            for alias in u.aliases:
                self._name_to_unit.setdefault(_norm(alias), u)
        self._all_names = [u.name for u in self.units]
        logger.debug("KnowledgeBase initialized: %d units (version=%s)", len(self.units), self.version)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.units)

    def __iter__(self):
        return iter(self.units)

    @property
    def names(self) -> list[str]:
        return list(self._all_names)

    def get(self, name: str) -> KnowledgeUnit | None:
        return self._name_to_unit.get(_norm(name))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and _norm(name) in self._name_to_unit

    # ------------------------------------------------------------------
    # Fuzzy matching used during evaluation (see Section 3.3 of the paper)
    # ------------------------------------------------------------------
    def fuzzy_match(
        self,
        candidate: str,
        threshold: float = 85.0,
    ) -> KnowledgeUnit | None:
        """Map a free-form string to a canonical knowledge unit.

        Uses normalized Indel similarity, matching the procedure described in
        Section 3.3 of the paper.  Returns ``None`` if the best score is below
        ``threshold``.
        """
        # Exact / alias hit comes first - fuzzy matching is the fallback.
        hit = self.get(candidate)
        if hit is not None:
            return hit

        if process is None:
            logger.warning(
                "rapidfuzz is not installed; falling back to exact match only. "
                "Run `pip install rapidfuzz` to enable fuzzy matching."
            )
            return None

        best = process.extractOne(
            candidate,
            self._all_names,
            scorer=fuzz.ratio,
            score_cutoff=threshold,
        )
        if best is None:
            return None
        return self.get(best[0])

    def fuzzy_match_many(
        self,
        candidates: Iterable[str],
        threshold: float = 85.0,
    ) -> list[KnowledgeUnit]:
        seen: set[str] = set()
        out: list[KnowledgeUnit] = []
        for c in candidates:
            u = self.fuzzy_match(c, threshold=threshold)
            if u is None or u.id in seen:
                continue
            seen.add(u.id)
            out.append(u)
        return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_units_file(path: str | os.PathLike[str]) -> tuple[list[KnowledgeUnit], str]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    version = payload.get("version", "0.0.0")
    raw = payload["knowledge_units"]
    units = []
    for entry in raw:
        if isinstance(entry, str):
            units.append(KnowledgeUnit(id=entry, name=entry))
        else:
            units.append(
                KnowledgeUnit(
                    id=entry.get("id", entry["name"]),
                    name=entry["name"],
                    domain=entry.get("domain", "unknown"),
                    aliases=list(entry.get("aliases", [])),
                )
            )
    return units, version


def load_knowledge_units(path: str | os.PathLike[str] | None = None) -> KnowledgeBase:
    """Load the SciCustom knowledge units.

    If ``path`` is omitted we look in the following order:
    1. ``SCICUSTOM_KB_PATH`` environment variable.
    2. The packaged ``assets/knowledge_units.json`` file.
    3. A file at ``./assets/knowledge_units.json`` relative to the cwd
       (useful for running from a source checkout without installing).
    """
    if path is None:
        env_path = os.environ.get("SCICUSTOM_KB_PATH")
        if env_path:
            path = env_path
            logger.debug("Using KB path from $SCICUSTOM_KB_PATH: %s", path)

    if path is None:
        try:
            # The data file lives outside the package itself (it ships as a
            # top-level asset directory), so the importlib lookup only works
            # for installs that include it via setuptools data_files.  We
            # fall back to a path search in that case.
            with resources.files("scicustom").joinpath("../assets/knowledge_units.json").open("r") as f:
                payload = json.load(f)
            units, version = _payload_to_units(payload)
            return KnowledgeBase(units=units, version=version)
        except (FileNotFoundError, ModuleNotFoundError):
            pass

    if path is None:
        # Walk up from this file to find the assets directory.  Works for
        # editable installs and source checkouts.
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.normpath(os.path.join(here, "..", "assets", "knowledge_units.json"))
        if os.path.exists(candidate):
            path = candidate

    if path is None:
        # Last resort: look at the cwd.
        cwd_candidate = os.path.join(os.getcwd(), "assets", "knowledge_units.json")
        if os.path.exists(cwd_candidate):
            path = cwd_candidate

    if path is None:
        raise FileNotFoundError(
            "Could not locate knowledge_units.json.  Set $SCICUSTOM_KB_PATH "
            "or pass `path=` explicitly."
        )

    units, version = _read_units_file(path)
    logger.info("Loaded %d knowledge units (version %s) from %s", len(units), version, path)
    return KnowledgeBase(units=units, version=version)


def _payload_to_units(payload: dict) -> tuple[list[KnowledgeUnit], str]:
    version = payload.get("version", "0.0.0")
    raw = payload["knowledge_units"]
    units: list[KnowledgeUnit] = []
    for entry in raw:
        if isinstance(entry, str):
            units.append(KnowledgeUnit(id=entry, name=entry))
        else:
            units.append(
                KnowledgeUnit(
                    id=entry.get("id", entry["name"]),
                    name=entry["name"],
                    domain=entry.get("domain", "unknown"),
                    aliases=list(entry.get("aliases", [])),
                )
            )
    return units, version


def select_relevant_units(
    kb: KnowledgeBase,
    ranked_names: Sequence[str],
    top_k: int = 10,
    fuzzy_threshold: float = 85.0,
) -> list[KnowledgeUnit]:
    """Take an ordered list of tag names and project it onto the KB.

    Used by the voting module to deduplicate names returned by multiple LLM
    judges into a single ranked subset of canonical knowledge units.
    """
    out: list[KnowledgeUnit] = []
    seen: set[str] = set()
    for name in ranked_names:
        unit = kb.fuzzy_match(name, threshold=fuzzy_threshold)
        if unit is None:
            continue
        if unit.id in seen:
            continue
        seen.add(unit.id)
        out.append(unit)
        if len(out) >= top_k:
            break
    return out

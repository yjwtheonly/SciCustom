"""Inference wrapper around the released ``LiamGu/SciCustom_Tagger`` model.

The tagger is a LLaMa-3-8B checkpoint finetuned to map a natural-language
scientific query to a subset of the SciCustom knowledge units.  It is the only
component of the framework that we ship as model weights; everything else in
this repo runs against frontier LLM APIs.

Two backends are supported:

* ``vllm`` (default) - matches the inference engine we used in the paper.  Best
  for batch tagging large corpora.
* ``hf`` - HuggingFace ``transformers`` fallback for single-query use, mostly
  useful when vLLM is unavailable (e.g., CPU-only laptop).
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

from scicustom.kb import KnowledgeBase, KnowledgeUnit, load_knowledge_units
from scicustom.prompts import TAGGER_SYSTEM, TAGGER_USER


logger = logging.getLogger(__name__)


DEFAULT_MODEL_ID = "LiamGu/SciCustom_Tagger"
DEFAULT_MAX_TOKENS = 256


# A handful of separators we have seen the model emit.  We strip them before
# fuzzy-matching against the knowledge base.
_TAG_SPLIT = re.compile(r"[,;\n]| - ")
_BULLET_LEAD = re.compile(r"^\s*([\-\*•]|\d+\.)\s*")
_QUOTE_TRIM = re.compile(r"^[\s\"'`]+|[\s\"'`.]+$")


@dataclass
class TagResult:
    """Output of a single tagging call."""

    query: str
    raw_output: str
    tags: list[KnowledgeUnit]

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.tags]

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "tags": [{"id": t.id, "name": t.name, "domain": t.domain} for t in self.tags],
            "raw_output": self.raw_output,
        }


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class _VLLMBackend:
    def __init__(
        self,
        model_id: str,
        *,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.85,
        **kwargs,
    ):
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except ImportError as exc:  # pragma: no cover - import error path
            raise ImportError(
                "vLLM is required for the default tagger backend.  Install it "
                "with `pip install vllm` or pass backend='hf' to use the "
                "transformers fallback."
            ) from exc

        self._SamplingParams = SamplingParams
        logger.info("Loading tagger weights from %s with vLLM (tp=%d, dtype=%s)",
                    model_id, tensor_parallel_size, dtype)
        t0 = time.time()
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            **kwargs,
        )
        logger.info("vLLM ready in %.1fs", time.time() - t0)
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, prompts: Sequence[str], *, max_tokens: int, temperature: float) -> list[str]:
        params = self._SamplingParams(
            temperature=temperature,
            top_p=1.0 if temperature == 0.0 else 0.95,
            max_tokens=max_tokens,
            stop=["<|eot_id|>", "<|end_of_text|>"],
        )
        outputs = self.llm.generate(list(prompts), params, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]


class _HFBackend:
    """A simple HuggingFace ``transformers`` fallback.

    Performance is significantly lower than the vLLM backend; this is intended
    for sanity checks and CPU-only experiments.
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: str | None = None,
        dtype: str = "bfloat16",
        **kwargs,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype, torch.bfloat16)

        logger.info("Loading tagger weights from %s with HF transformers (device=%s)", model_id, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            **kwargs,
        ).to(device)
        self.model.eval()

    def generate(self, prompts: Sequence[str], *, max_tokens: int, temperature: float) -> list[str]:
        import torch

        outputs: list[str] = []
        for prompt in prompts:
            enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            text = self.tokenizer.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            outputs.append(text)
        return outputs


# ---------------------------------------------------------------------------
# Tagger
# ---------------------------------------------------------------------------

class SciTagger:
    """High-level interface for the released SciCustom tagger.

    Example:
        >>> tagger = SciTagger.from_pretrained()
        >>> tagger.tag("How do alcohols cause sleep disorders?")
        [KnowledgeUnit(id='ku_017', name='Alcohol', ...),
         KnowledgeUnit(id='ku_539', name='Sleep disorder', ...), ...]
    """

    def __init__(
        self,
        backend,
        knowledge_base: KnowledgeBase,
        *,
        fuzzy_threshold: float = 85.0,
        max_tags: int = 8,
    ):
        self._backend = backend
        self.kb = knowledge_base
        self.fuzzy_threshold = fuzzy_threshold
        self.max_tags = max_tags
        # Cached chat template formatter to avoid re-resolving it per call.
        self._tokenizer = getattr(backend, "tokenizer", None)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        backend: str = "vllm",
        knowledge_base: KnowledgeBase | str | None = None,
        fuzzy_threshold: float = 85.0,
        max_tags: int = 8,
        **backend_kwargs,
    ) -> "SciTagger":
        if backend == "vllm":
            be = _VLLMBackend(model_id, **backend_kwargs)
        elif backend in ("hf", "transformers"):
            be = _HFBackend(model_id, **backend_kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend!r} (expected 'vllm' or 'hf')")

        if knowledge_base is None or isinstance(knowledge_base, (str, os.PathLike)):
            kb = load_knowledge_units(knowledge_base)
        else:
            kb = knowledge_base

        return cls(be, kb, fuzzy_threshold=fuzzy_threshold, max_tags=max_tags)

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------
    def _format(self, query: str) -> str:
        user_msg = TAGGER_USER.format(query=query)
        if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": TAGGER_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

        # Bare-bones llama-3 chat template, used only when no tokenizer is
        # available (e.g., when a downstream user constructs the tagger
        # manually for tests).
        return (
            "<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n{TAGGER_SYSTEM}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def tag(
        self,
        query: str | Iterable[str],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> list[KnowledgeUnit] | list[TagResult]:
        """Tag a single query or a batch.

        For a single string we return the list of matched knowledge units.
        For a batch we return one :class:`TagResult` per query so callers can
        inspect the raw model output.
        """
        if isinstance(query, str):
            return self._tag_batch([query], max_tokens=max_tokens, temperature=temperature)[0].tags
        queries = list(query)
        return self._tag_batch(queries, max_tokens=max_tokens, temperature=temperature)

    def _tag_batch(
        self,
        queries: Sequence[str],
        *,
        max_tokens: int,
        temperature: float,
    ) -> list[TagResult]:
        if not queries:
            return []

        prompts = [self._format(q) for q in queries]
        logger.debug("Tagging batch of %d queries", len(queries))
        t0 = time.time()
        raw_outputs = self._backend.generate(prompts, max_tokens=max_tokens, temperature=temperature)
        dt = time.time() - t0
        if dt > 0:
            logger.info("Tagged %d queries in %.2fs (%.1f q/s)", len(queries), dt, len(queries) / dt)

        results: list[TagResult] = []
        for q, raw in zip(queries, raw_outputs):
            tags = self._parse_and_match(raw)
            results.append(TagResult(query=q, raw_output=raw, tags=tags))
        return results

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------
    def _parse_and_match(self, raw: str) -> list[KnowledgeUnit]:
        # Trim chat scaffolding the model sometimes echoes back.
        for marker in ("Answer:", "Tags:", "Knowledge units:"):
            idx = raw.lower().find(marker.lower())
            if idx >= 0:
                raw = raw[idx + len(marker):]
                break

        pieces = _TAG_SPLIT.split(raw)
        candidates: list[str] = []
        for piece in pieces:
            piece = _BULLET_LEAD.sub("", piece)
            piece = _QUOTE_TRIM.sub("", piece)
            if not piece:
                continue
            # Some checkpoints occasionally emit "Alcohol (Chemistry)"; strip
            # the trailing parenthetical for fuzzy matching.
            piece = re.sub(r"\s*\([^)]*\)\s*$", "", piece)
            candidates.append(piece)

        matched = self.kb.fuzzy_match_many(candidates, threshold=self.fuzzy_threshold)
        return matched[: self.max_tags]


# A tiny module-level helper that the README uses in its quickstart.  It
# defaults to the HuggingFace backend so that a cold install on a fresh
# machine still works without vLLM.
def quick_tag(query: str, model_id: str = DEFAULT_MODEL_ID) -> list[str]:
    tagger = SciTagger.from_pretrained(model_id, backend="hf")
    return [u.name for u in tagger.tag(query)]

# SciCustom

Official implementation of **SciCustom: A Framework for Custom Evaluation of Scientific Capabilities in Large Language Models**.

SciCustom is an ontology-driven framework that automatically constructs *custom* benchmarks for evaluating LLMs on user-specified scientific capabilities, without expert annotation or synthetic question generation. The framework organizes scattered scientific data into ontology-grounded *knowledge units*, then composes those units into a benchmark tailored to any evaluation requirement (e.g. "technical chemistry", "human aging", or a novel application such as "pericyclic reaction").

```
                Offline                                Online
+-----------------+   +---------+    +-------------+   +---------------------+
| Ontology DAGs   |-->| Granularity-|-->| 642 Knowledge|-->| Requirement parsing |
| (OBO/BioPortal) |   | based DFS   |   |    Units    |   | + multi-judge voting|
+-----------------+   +---------+    +------+------+   +----------+----------+
                                            v                       v
        +--------------+      +-----------+----------+   +----------+----------+
        | Tagger (8B)  |----->| Tagged scientific QA |-->| Binary-search retrieval|
        +--------------+      +----------+-----------+   | + proxy subset (K2)   |
                                         |              +-----------+-----------+
                                         v                          v
                                  +-----------------+      +--------+---------+
                                  | MCQ transformation |<---| Selected subset  |
                                  +-----------------+      +------------------+
                                                                    v
                                                            Custom Benchmark
```

The tagger checkpoint is released at <https://huggingface.co/LiamGu/SciCustom_Tagger>.

---

## Installation

We recommend Python 3.10+. Install from source:

```bash
git clone https://github.com/yjwtheonly/SciCustom.git
cd SciCustom
pip install -e .
```

For tagger inference you also need `vllm` (or `transformers` for the slower CPU fallback). Set the API keys you intend to use:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
```

## Quick start

### 1. Tag a scientific query with the released model

```python
from scicustom import SciTagger

tagger = SciTagger.from_pretrained("LiamGu/SciCustom_Tagger")
print(tagger.tag("How do alcohols cause sleep disorders?"))
# -> [KnowledgeUnit(name='Alcohol', ...),
#     KnowledgeUnit(name='Sleep disorder', ...),
#     KnowledgeUnit(name='Mental disease', ...)]
```

For batch tagging a corpus:

```bash
python scripts/tag_queries.py \
    --input data/sciriff.jsonl \
    --output runs/sciriff.tagged.jsonl \
    --batch-size 64
```

### 2. Build a custom benchmark

Given a tagged corpus and a YAML config that describes the requirement:

```bash
python scripts/build_benchmark.py \
    --config configs/chemistry/pericyclic_reaction.yaml \
    --corpus runs/sciriff.tagged.jsonl \
    --out-dir runs/pericyclic_reaction
```

The pipeline writes the following artifacts:

| File | Contents |
| --- | --- |
| `voting.json` | Top-K knowledge units selected by the multi-judge vote |
| `retrieved.jsonl` | Pre-cutoff candidate set |
| `proxy.jsonl` | K2-sized proxy subset (SubLIME-style) |
| `benchmark.jsonl` | Final MCQs ready for evaluation |
| `config.json` | A snapshot of the run configuration |

### 3. Evaluate an LLM on the benchmark

```bash
python scripts/evaluate_models.py \
    --benchmark runs/pericyclic_reaction/benchmark.jsonl \
    --models configs/eval_models.yaml \
    --out runs/pericyclic_reaction/eval.json
```

## Project structure

```
SciCustom/
├── scicustom/
│   ├── tagger.py         # Inference for the released LiamGu/SciCustom_Tagger
│   ├── kb.py             # Knowledge base loading and fuzzy matching
│   ├── ontology.py       # Algorithm 1: granularity-based DFS
│   ├── voting.py         # Section 2.4 multi-judge voting
│   ├── retrieval.py      # Algorithm 2 binary search + proxy subset selection
│   ├── mcq.py            # MCQ transformation
│   ├── eval.py           # LLM evaluation + rank correlation helpers
│   ├── llm.py            # OpenAI / Anthropic / Gemini chat backends
│   ├── prompts.py        # All prompt templates (Appendix G)
│   ├── pipeline.py       # End-to-end glue
│   └── utils.py          # IO + logging helpers
├── scripts/              # CLI entry points (tagging, building, evaluation)
├── configs/              # Per-requirement YAML configs
├── assets/
│   └── knowledge_units.json  # 642 ontology-grounded knowledge units
├── examples/
└── tests/
```

## Knowledge units

The 642 knowledge units (641 scientific + 1 `Non-Scientific`) are stored in `assets/knowledge_units.json`. They were obtained by running the granularity-based DFS (Algorithm 1) over 227 scientific subdiscipline DAGs aggregated from OBO, BioPortal and OLS. The same vocabulary is used by the released tagger (`LiamGu/SciCustom_Tagger`) and the voting / retrieval modules; if you want to use a custom vocabulary, pass `--knowledge-units path/to/your.json` to the relevant scripts or `SciTagger.from_pretrained(..., knowledge_base=path)` in code.

## Reproducing the paper

| Section | Command |
| --- | --- |
| 2.2  Knowledge unit selection (Alg. 1) | `python scripts/select_knowledge_units.py --dag <dag.json> --model gpt-5 --out my_kus.json` |
| 2.3  Tagger training | Not included in this release; see `LiamGu/SciCustom_Tagger` for the released checkpoint. |
| 2.3  Tagger inference | `python scripts/tag_queries.py ...` (see above) |
| 2.4  Voting | covered by `scripts/build_benchmark.py` |
| 2.5  Binary search + proxy selection | covered by `scripts/build_benchmark.py` |
| 2.5  MCQ transformation | covered by `scripts/build_benchmark.py` |
| 3.1  Ranking consistency | `python scripts/evaluate_models.py` then aggregate accuracies with the helpers in `scicustom.eval` |

The supervised data used to train the tagger (50K synthetic + 30K real queries) was produced by the prompt templates in `scicustom/prompts.py` (`QUERY_GEN_LOW` / `QUERY_GEN_HIGH` with the personae list). We do not redistribute the training data because most of the underlying instruction-tuning corpora carry separate licenses; the templates are sufficient to regenerate it from the released sources (SciRIFF, SciInstruct, Mol-Instruct, MultiMedQA, SciEval, MMLU-Pro, GPQA, IfBench, SimpleQA).

## Limitations

See Section "Limitations" of the paper. In short, coverage is bounded by the underlying ontologies (predominantly biomedical and chemical) and by the source corpus `D`. The tagger inherits both biases. Adding a new ontology requires re-running Algorithm 1 and re-training the tagger.

## Citation

```bibtex
@inproceedings{gu2025scicustom,
  title     = {SciCustom: A Framework for Custom Evaluation of Scientific Capabilities in Large Language Models},
  author    = {Gu, Yiyang and Yang, Junwei and Luo, Junyu and Yuan, Ye and Feng, Bin and Xia, Yingce and Xie, Shufang and Liu, Kaili and Wu, Bohan and Shi, Qi and Li, Haoran and Xiao, Beier and Xiao, Zhiping and Luo, Xiao and Zhang, Weizhi and Yu, Philip S. and Liu, Zequn and Zhang, Ming},
  year      = {2025},
}
```

## License

Code is released under the Apache 2.0 License (see `LICENSE`). The 642 knowledge units are aggregated from publicly licensed ontologies (OBO, BioPortal, OLS); please honor their individual licenses if you redistribute the JSON.

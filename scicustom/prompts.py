"""Prompt templates used across the SciCustom pipeline.

The templates here mirror those reported in Appendix G of the paper. They are
collected in one place so that downstream tweaks (rephrasing for a new model
family, swapping in a different persona list, ...) only require edits to this
file.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Ontology granularity classification (Algorithm 1, "label <- LLM classifies v")
# ---------------------------------------------------------------------------

GRANULARITY_SYSTEM = "You are a helpful AI assistant."

GRANULARITY_USER = textwrap.dedent(
    """\
    Determine whether the given term is a suitable scientific subfield and
    answer with one of the following categories:

    (moderate): The term is appropriately specific for a scientific subfield.
    It refers to a category that is neither too general nor too specific for
    its scientific context. Its scale is similar to the scale of chapter
    names in subject textbooks.

    (too coarse): The term is overly broad or vague for a scientific
    subfield. It encompasses a wide range of concepts that could be divided
    into smaller, more specific subfields.

    (too fine): The term is overly specific and pertains to a very narrow
    aspect of a scientific subfield. It may be too detailed to serve as a
    broader category within the discipline.

    Answer at the beginning, explain later. The examples are as follows:

    Example 1:
    Input: term: anatomical entity
    Output: (moderate); Explanation: Anatomical entity refers to a category
    that is sufficiently specific for many biological subfields but not too
    narrow.

    Example 2:
    Input: term: nuclear structure
    Output: (moderate); Explanation: Nuclear structure is a well-defined
    category in molecular biology, specific but not too narrow.

    Example 3:
    Input: term: electronic file status
    Output: (too fine); Explanation: The term refers to a very specific
    technical concept that is too detailed to be considered a scientific
    subfield.

    Example 4:
    Input: term: b-lymphocyte
    Output: (too fine); Explanation: While important, the term refers to a
    very specific type of cell, not a broad enough category to encompass a
    subfield.

    Example 5:
    Input: term: continuant
    Output: (too coarse); Explanation: The term is too general and could
    refer to a wide variety of objects or concepts, making it too broad for
    a specific subfield.

    Example 6:
    Input: term: occurrent
    Output: (too coarse); Explanation: Occurrent is overly vague and applies
    to many concepts, making it too broad for a scientific subfield.

    Input: term: {term}.
    """
)


# ---------------------------------------------------------------------------
# Query generation for tagger training (kept here so the offline pipeline can
# reproduce the synthetic corpus we used to fine-tune the released tagger).
# ---------------------------------------------------------------------------

QUERY_GEN_SYSTEM = "You are a helpful AI assistant."

QUERY_GEN_LOW = (
    "You are a {persona}. Generate a user query containing the following "
    "keywords: {keywords}. Do not introduce other scientific entities or "
    "topics. Only return the query."
)

QUERY_GEN_HIGH = (
    "You are a {persona}. Generate a user query containing the following "
    "keywords: {keywords}. Do not introduce other scientific entities or "
    "topics. Make the query long and complex. Only return the query."
)

# 20 personae, matches Appendix G.
SCIENTIFIC_PERSONAS = [
    "Astrophysicist",
    "Marine Biologist",
    "AI Researcher",
    "Molecular Geneticist",
    "Quantum Physicist",
    "Environmental Chemist",
    "Neuroscientist",
    "Ecologist",
    "Bioinformatician",
    "Pharmacologist",
    "Geologist",
    "Biomedical Engineer",
    "Mathematical Modeler",
    "Virologist",
    "Behavioral Psychologist",
    "Data Scientist",
    "Theoretical Chemist",
    "Climate Scientist",
    "Structural Biologist",
    "Robotics Engineer",
]


# ---------------------------------------------------------------------------
# Benchmark requirement descriptions (used as auxiliary context for voting and
# MCQ generation).  These mirror the per-task descriptions in Appendix G; they
# are loaded from configs/<domain>.yaml in normal use, but the defaults below
# act as a safety net so the package is self-contained even when the configs
# directory is missing.
# ---------------------------------------------------------------------------

BENCHMARK_DESCRIPTIONS: dict[str, str] = {
    "analytical_chemistry": (
        "Generate questions that test knowledge and reasoning in analytical "
        "chemistry.  The questions should assess understanding of how "
        "experimental analytical signals (e.g., NMR, IR, UV-Vis, mass "
        "spectra, chromatographic behavior, titration curves) relate to "
        "molecular structure, composition, concentration, or purity.  Focus "
        "on conceptual interpretation and chemical reasoning rather than "
        "numerical data processing or instrument-specific operating "
        "procedures."
    ),
    "inorganic_chemistry": (
        "Generate questions that test core knowledge and reasoning in "
        "inorganic chemistry.  The questions should focus on electronic "
        "structure, oxidation states, coordination geometry, ligand field "
        "effects, symmetry, and periodic trends in inorganic systems.  "
        "Emphasize conceptual understanding of structure-property "
        "relationships rather than memorization of isolated facts."
    ),
    "material_science": (
        "Generate questions that evaluate understanding in materials "
        "science.  The questions should assess how atomic or "
        "microstructural features (e.g., crystal structure, defects, "
        "phases, interfaces) determine macroscopic properties such as "
        "mechanical strength, electrical conductivity, or thermal "
        "behavior.  Focus on structure-property reasoning rather than "
        "detailed synthesis protocols."
    ),
    "organic_chemistry": (
        "The organic chemistry benchmark assesses a wide range of skills "
        "on reasoning about chemical structures and reaction pathways, "
        "such as Reaction Mechanism Identification, Product Prediction, "
        "NMR Signal Prediction, Number of Isomers, Polymer Chemistry, "
        "Nomenclature Conversion and Organic Reactivity."
    ),
    "physical_chemistry": (
        "Generate questions that test conceptual understanding in physical "
        "chemistry.  The questions should assess reasoning about "
        "thermodynamics, kinetics, equilibrium, and molecular-level "
        "physical principles.  Emphasize qualitative reasoning about "
        "trends and relationships rather than explicit numerical "
        "calculation."
    ),
    "technical_chemistry": (
        "Generate questions that assess reasoning in technical and "
        "industrial chemistry.  The questions should focus on chemical "
        "processes at scale, such as reactor behavior, process "
        "optimization, safety considerations, and material or energy "
        "conversion.  Emphasize reasoning about system-level behavior "
        "rather than detailed engineering design."
    ),
    "virology": (
        "Generate questions that test conceptual understanding in "
        "virology.  The questions should assess knowledge of viral "
        "structure, replication cycles, genome organization, and "
        "interactions with host cells and immune systems.  Avoid clinical "
        "treatment guidelines or laboratory diagnostic protocols."
    ),
    "human_aging": (
        "Generate questions that probe understanding of biological "
        "mechanisms of human aging.  The questions should focus on "
        "molecular, cellular, and systemic processes associated with "
        "aging, such as genomic stability, cellular senescence, metabolic "
        "regulation, and tissue-level decline.  Emphasize mechanistic "
        "reasoning rather than epidemiological statistics."
    ),
    "medical_genetics": (
        "Generate questions that test reasoning in medical genetics.  The "
        "questions should assess understanding of inheritance patterns, "
        "penetrance, and genotype-phenotype relationships, and genetic "
        "variation.  Focus on conceptual genetic reasoning rather than "
        "clinical decision-making."
    ),
    "anatomy": (
        "Generate questions that evaluate knowledge of human anatomy.  The "
        "questions should focus on the identification, spatial "
        "relationships, and functional roles of anatomical structures.  "
        "Avoid surgical procedures or pathological conditions."
    ),
    "nutrition": (
        "Generate questions that assess understanding of nutritional "
        "science.  The questions should focus on the biological roles of "
        "macro- and micronutrients, their involvement in metabolism, and "
        "the physiological consequences of deficiency or imbalance.  "
        "Emphasize mechanistic understanding over dietary "
        "recommendations."
    ),
}


# ---------------------------------------------------------------------------
# Voting-based relevant tag selection (Section 2.4)
# ---------------------------------------------------------------------------

VOTING_SYSTEM = (
    "You are an expert in {domain}.  Your task is to map a benchmark "
    "description to the most relevant technical tags"
)

VOTING_USER = textwrap.dedent(
    """\
    Task:
    Given a {domain} benchmark description, identify and rank the most
    relevant tags from a candidate list.

    Benchmark Description: {description}

    Candidate Tags (sorted by frequency, lowest frequency first; lower
    frequency usually indicates higher specificity): {tag_list}

    Ranking Principles: Rank tags from highest to lowest relevance to the
    benchmark, following these rules:

    Relevance First:
    A tag is relevant if it directly reflects the core concepts, tasks, data
    modalities, or evaluation focus of the benchmark.  Irrelevant or weakly
    related tags should not be selected.

    Specificity as a Tie-breaker:
    If multiple tags are similarly relevant, rank the more specific and
    narrowly scoped tag higher.  Prefer concrete technical terms (e.g.,
    "Histone Acetylation Prediction") over broader categories (e.g.,
    "Epigenetics").

    Avoid Overly Generic Tags:
    High-level or generic tags (e.g., "biological process", "chemical
    entity") should only be selected if no more specific alternative
    applies.

    Frequency Awareness:
    When relevance and specificity are comparable, prefer lower-frequency
    tags, as they tend to be more precise.

    Output Requirements:
    Return a single list of tags, sorted from most to least relevant.  For
    efficiency, return only the top 100 tags (or fewer if fewer are
    relevant).  Do not include explanations, scores, or extra text---output
    the ranked list only.
    """
)


# ---------------------------------------------------------------------------
# Benchmark generation prompt - used by the fully synthetic GPT-5 baseline
# reported in Appendix C.  We keep it next to the SciCustom prompts so the
# baseline is reproducible from the same module.
# ---------------------------------------------------------------------------

BENCHMARK_GEN_SYSTEM = (
    "You are an expert in {domain} and tasked with constructing a "
    "high-quality benchmark to assess the domain-specific knowledge "
    "abilities of large language models.  Please return the benchmark in "
    "a JSON format."
)

BENCHMARK_GEN_USER = textwrap.dedent(
    """\
    Your task is to generate exactly {k} single-choice questions in the
    domain of {domain}.
    Detailed description of this domain: {description}

    The questions should:
    1. Focus on core concepts, expert-level knowledge, and non-trivial
       reasoning in this domain.
    2. Avoid trivial questions, purely factual memorization, or overly
       ambiguous questions.
    3. Include a mix of:
       - Conceptual understanding
       - Mechanism or principle-based reasoning
       - Application or scenario-based reasoning
    4. Be answerable without external tools, but not solvable by
       surface-level pattern matching.

    Question format:
    1. Each question must have 4-5 options.
    2. Options should be concise and mutually exclusive.
    3. Each question must have only one correct answers.

    Output format (STRICT):
    Return only a JSON array of length {k}.
    Each element must have the following structure:
    {{
      "query": "<question text with options labeled A, B, C, D (and E if applicable)>",
      "answer": "<correct option label>"
    }}
    """
)


# ---------------------------------------------------------------------------
# MCQ transformation (Section 2.5, "Data-Grounded Benchmark Generation")
# ---------------------------------------------------------------------------

MCQ_SYSTEM = (
    "You are an expert in {domain} and tasked with curating a rigorous "
    "benchmark to evaluate the capabilities of Large Language Models.  "
    "Please return the processed entry in a JSON format."
)

MCQ_USER = textwrap.dedent(
    """\
    Your task is to convert the following raw problem content into a
    standardized single-choice question suitable for LLM evaluation.
    Raw problem: {input_content}

    Conversion Guidelines:
    1. Format Adaptation:
       - If the input is already a multiple-choice question: Preserve the
         original stem and options exactly.  Ensure the formatting aligns
         with the output requirements.
       - If the input is not a multiple-choice question: Convert it into a
         single-choice question by generating 3-4 incorrect options
         (distractors).
    2. Distractor Engineering:
       - Avoid trivial errors, logical fallacies that are easily filtered,
         or clearly unrelated concepts.
    3. Fidelity & Difficulty:
       - Strict adherence to the factual truth and reasoning logic of the
         original content is required.
       - Do not simplify the problem complexity.  The resulting MCQ must
         maintain the same discriminative power as the original input.
    4. Exclusivity: Ensure there is exactly one indisputably correct
       option.

    Question format:
    1. The final output must contain 4-5 options (A, B, C, D, [E]).
    2. Options should be concise and mutually exclusive.

    Output format (STRICT):
    Return only a single JSON object.
    The object must have the following structure:
    {{
      "query": "<question stem followed by options labeled A, B, C, D (and E if applicable), separated by newlines>",
      "answer": "<correct option label, e.g., `A`>"
    }}
    """
)


# ---------------------------------------------------------------------------
# Tagger prompt (used at inference time by the released LiamGu/SciCustom_Tagger
# checkpoint).  The model was finetuned with this exact template; changing it
# in the wild will hurt recall.
# ---------------------------------------------------------------------------

TAGGER_SYSTEM = (
    "You are SciCustom Tagger.  Given a scientific user query, output a "
    "concise list of the most relevant ontology-grounded knowledge units "
    "from the SciCustom knowledge unit set.  If the query is not a "
    "scientific query, output `Non-Scientific`."
)

TAGGER_USER = "Query: {query}\n\nReturn a comma-separated list of knowledge units."


@dataclass
class PromptTemplate:
    """Tiny wrapper for system + user prompt pairs.

    We use this for the LLM clients to avoid passing two strings around.
    """

    system: str
    user: str

    def format(self, **kwargs) -> tuple[str, str]:
        return self.system.format(**kwargs), self.user.format(**kwargs)

"""SciCustom: Custom evaluation of scientific capabilities in LLMs.

See `https://github.com/yjwtheonly/SciCustom` for project documentation.
"""

from scicustom._version import __version__
from scicustom.tagger import SciTagger
from scicustom.kb import KnowledgeBase, load_knowledge_units
from scicustom.pipeline import build_benchmark

__all__ = [
    "__version__",
    "SciTagger",
    "KnowledgeBase",
    "load_knowledge_units",
    "build_benchmark",
]

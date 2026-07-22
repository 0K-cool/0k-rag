"""
0K-RAG - 100% Local RAG System with Hybrid Search

A production-ready Retrieval-Augmented Generation (RAG) system with:
- Contextual chunking (Llama 3.1 8B)
- Vector search (nomic-embed-text)
- BM25 keyword search
- Reciprocal Rank Fusion (RRF)
- BGE reranking (Apple Silicon GPU)
- PII sanitization
- Multi-project support

Author: Kelvin Lomboy
License: MIT
Version: 1.0.0
"""

# Single source of truth is pyproject.toml; read it at import so this can never
# drift from the packaged version again (it was stuck at 1.0.0 for 5 releases).
try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError
    try:
        __version__ = _pkg_version("0k-rag")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"
except ImportError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__author__ = "Kelvin Lomboy"
__license__ = "MIT"

from rag.indexing.indexer import KnowledgeBaseIndexer
from rag.retrieval.pipeline import RetrievalPipeline

__all__ = [
    "KnowledgeBaseIndexer",
    "RetrievalPipeline",
]

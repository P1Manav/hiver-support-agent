"""
src/retrieval/__init__.py
──────────────────────────
FAISS-based retrieval package.
"""
from .indexer import build_index, load_index
from .retriever import Retriever

__all__ = ["build_index", "load_index", "Retriever"]

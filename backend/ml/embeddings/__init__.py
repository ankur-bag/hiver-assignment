"""
Sentence Transformer Embedding Module.
Provides lazy-loaded, thread-safe embeddings with L2 normalization.
"""

from .encoder import EmbeddingEncoder, get_embedding_encoder

__all__ = ["EmbeddingEncoder", "get_embedding_encoder"]

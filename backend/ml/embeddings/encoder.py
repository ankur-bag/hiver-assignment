"""
Sentence Transformer Encoder Wrapper.
Handles lazy loading of the multilingual Sentence Transformer model,
generating normalized embeddings matching training conditions.
"""

import logging
from threading import Lock
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from ml.config import config

logger = logging.getLogger(__name__)


class EmbeddingEncoder:
    """
    Singleton wrapper around SentenceTransformer to ensure model is loaded
    only once in memory and reused across all prediction requests.
    """
    _instance = None
    _lock = Lock()

    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        logger.info(f"Loading SentenceTransformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.dimension = config.EMBEDDING_DIMENSION
        logger.info(f"SentenceTransformer model loaded successfully. Dim: {self.dimension}")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Generates L2-normalized embeddings for given text(s).
        
        Args:
            texts: Single string or list of strings.
            batch_size: Batch size for encoding multiple texts.
            show_progress_bar: Whether to display encoding progress.
            
        Returns:
            np.ndarray of shape (N, 384) or (384,) with L2-normalized floats.
        """
        if isinstance(texts, str):
            single = True
            text_list = [texts]
        else:
            single = False
            text_list = list(texts)

        embeddings = self.model.encode(
            text_list,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,  # Crucial: matches training normalize_embeddings=True
            convert_to_numpy=True
        )

        if single:
            return embeddings[0]
        return embeddings


_encoder_instance = None
_encoder_lock = Lock()


def get_embedding_encoder() -> EmbeddingEncoder:
    """Returns the shared singleton instance of EmbeddingEncoder."""
    global _encoder_instance
    if _encoder_instance is None:
        with _encoder_lock:
            if _encoder_instance is None:
                _encoder_instance = EmbeddingEncoder()
    return _encoder_instance

import logging
import gc
from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer

from configs import get_settings

logger = logging.getLogger(__name__)


def get_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")
    return device


class EmbeddingService:
    """
    Embedding service using sentence-transformers.
    
    Provides vector representations for text used in RAG retrieval.
    """

    _instance = None
    _model = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            settings = get_settings()
            self._device = get_device()
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            self._model = SentenceTransformer(
                settings.embedding_model,
                device=self._device
            )
            logger.info(f"Embedding model loaded on {self._device}")

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Parameters
        ----------
        text : str
            Text to embed

        Returns
        -------
        list[float]
            Embedding vector
        """
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Parameters
        ----------
        texts : list[str]
            Texts to embed

        Returns
        -------
        list[list[float]]
            List of embedding vectors
        """
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    @staticmethod
    def clear_cache():
        """Clear GPU/MPS memory cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Get cached embedding service instance."""
    return EmbeddingService()

import logging
from typing import List

from sentence_transformers import SentenceTransformer

from tagging_service.app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Bi-encoder embedding service using SentenceTransformers."""

    _instance = None

    def __init__(self):
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")

    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    def embed(self, texts: List[str] | str) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_one(self, text: str) -> List[float]:
        return self.embed(text)[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self.embed(texts)

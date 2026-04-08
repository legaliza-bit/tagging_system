import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers.cross_encoder import CrossEncoder

from app.config import settings

logger = logging.getLogger(__name__)


class RerankerService:
    _instance = None

    def __init__(self) -> None:
        self._model, self._meta = self._load_model()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    def _load_model(self) -> Tuple[CrossEncoder, dict]:
        ft_path = Path(settings.FINETUNED_MODEL_DIR)
        if ft_path.exists() and any(ft_path.iterdir()):
            meta_file = ft_path / "finetune_meta.json"
            meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
            logger.info(f"Loading fine-tuned cross-encoder from {ft_path}")
            return CrossEncoder(str(ft_path), max_length=meta.get("max_length", 256)), meta
        logger.info(
            f"No fine-tuned model at {ft_path}, "
            f"falling back to base model: {settings.RERANKER_BASE_MODEL}"
        )
        return CrossEncoder(settings.RERANKER_BASE_MODEL), {}

    def _fmt_doc(self, text: str) -> str:
        return text[:self._meta.get("doc_max_chars", 400)]

    def _fmt_tag(self, name: str) -> str:
        if self._meta.get("tag_format", "name+desc") == "name+desc":
            descs = self._meta.get("dbpedia_descriptions", {})
            desc = descs.get(name, "")
            return f"{name}: {desc}" if desc else name
        return name

    @staticmethod
    def _sigmoid(logits) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.array(logits, dtype=float)))

    def _score_pairs(self, query: str, candidates: List[str]) -> List[Tuple[str, float]]:
        if not candidates:
            return []
        pairs = [[query, c] for c in candidates]
        logits = self._model.predict(pairs, show_progress_bar=False)
        scores = self._sigmoid(logits).tolist()
        return sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    def score_pairs_raw(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Score arbitrary (a, b) pairs in input order. No sorting."""
        if not pairs:
            return []
        logits = self._model.predict(list(pairs), show_progress_bar=False)
        return self._sigmoid(logits).tolist()

    def rerank_tags_for_document(
        self, document_text: str, tag_names: List[str]
    ) -> List[Tuple[str, float]]:
        """Score document against each tag. Returns [(tag_name, score), ...] descending."""
        doc = self._fmt_doc(document_text)
        fmt_tags = [self._fmt_tag(t) for t in tag_names]
        raw = self._score_pairs(doc, fmt_tags)
        fmt_to_raw = dict(zip(fmt_tags, tag_names))
        return [(fmt_to_raw.get(ft, ft), score) for ft, score in raw]

    def rerank_tags_for_tag(
        self, new_tag_name: str, existing_tag_names: List[str]
    ) -> List[Tuple[str, float]]:
        """Deduplication: score new_tag against each existing tag. Returns descending."""
        fmt_existing = [self._fmt_tag(t) for t in existing_tag_names]
        raw = self._score_pairs(new_tag_name, fmt_existing)
        fmt_to_raw = dict(zip(fmt_existing, existing_tag_names))
        return [(fmt_to_raw.get(ft, ft), score) for ft, score in raw]

    def rerank(self, query: str, candidates: List[str]) -> List[Tuple[str, float]]:
        """Generic rerank for UI free-text search suggestions."""
        return self._score_pairs(query, candidates)

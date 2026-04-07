import logging
from typing import List, Tuple

from app.config import settings

logger = logging.getLogger(__name__)


class TagRetrievalService:
    """
    Vector search → DB fetch → rerank → score fusion (0.7 * reranker + 0.3 * vector).
    """

    def __init__(self, vector_store, repo, reranker):
        self.vector_store = vector_store
        self.repo = repo
        self.reranker = reranker

    async def retrieve(self, text: str, embedding: List[float]) -> List[Tuple]:
        hits = await self.vector_store.search_similar_tags(embedding, settings.TOP_K_CANDIDATES)
        if not hits:
            return []

        tag_ids = [h[0] for h in hits]
        vector_scores = {h[0]: h[1] for h in hits}

        db_tags = await self.repo.get_by_ids(tag_ids)
        if not db_tags:
            return []

        tag_texts = [
            f"{t.name}: {t.description}" if t.description else t.name
            for t in db_tags
        ]
        reranked = self.reranker.rerank_tags_for_document(text, tag_texts)

        name_to_tag = {
            (f"{t.name}: {t.description}" if t.description else t.name): t
            for t in db_tags
        }
        scored = []
        for fmt_name, rr_score in reranked:
            tag = name_to_tag.get(fmt_name)
            if not tag:
                continue
            score = 0.7 * rr_score + 0.3 * vector_scores.get(tag.id, 0.0)
            scored.append((tag, score))

        return sorted(scored, key=lambda x: x[1], reverse=True)

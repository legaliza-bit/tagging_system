import logging
from typing import List, Tuple

from app.config import settings

logger = logging.getLogger(__name__)


class TagRetrievalService:
    """
    Vector search → DB fetch → rerank → score fusion
    → relative scoring + frequency penalty.
    """

    def __init__(self, vector_store, repo, reranker):
        self.vector_store = vector_store
        self.repo = repo
        self.reranker = reranker

    async def retrieve(self, text: str, embedding: List[float]) -> List[Tuple]:
        hits = await self.vector_store.search_similar_tags(
            embedding, settings.TOP_K_CANDIDATES
        )
        if not hits:
            return []

        tag_ids = [h[0] for h in hits]
        vector_scores = {h[0]: h[1] for h in hits}

        db_tags = await self.repo.get_by_ids(tag_ids)
        if not db_tags:
            return []

        tag_names = [t.name for t in db_tags]
        reranked = self.reranker.rerank_tags_for_document(text, tag_names)

        name_to_tag = {t.name: t for t in db_tags}

        # Base score fusion
        scored = []
        for fmt_name, rr_score in reranked:
            tag = name_to_tag.get(fmt_name)
            if not tag:
                continue
            score = 0.7 * rr_score + 0.3 * vector_scores.get(tag.id, 0.0)
            scored.append((tag, score))

        if not scored:
            return []

        # Frequency penalty: adjusted = score - alpha * (tag_doc_count / total_docs)
        if settings.TAG_FREQ_ALPHA > 0:
            total_docs = await self.repo.count()
            if total_docs > 0:
                tag_doc_counts = await self.repo.get_tag_doc_counts()
                scored = [
                    (tag, score - settings.TAG_FREQ_ALPHA * (tag_doc_counts.get(tag.id, 0) / total_docs))
                    for tag, score in scored
                ]

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

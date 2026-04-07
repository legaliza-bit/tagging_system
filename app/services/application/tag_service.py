import logging
from typing import List, Optional

from app.config import settings
from app.db.repository import TagRepository
from app.schemas.tag import SimilarTag, TagCreateResult, TagResponse
from app.services.infrastructure.embedding import EmbeddingService
from app.services.infrastructure.reranker import RerankerService
from app.services.infrastructure.retrieval import TagRetrievalService
from app.services.infrastructure.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class TagService:

    def __init__(self, db):
        self.repo = TagRepository(db)
        self.db = db
        self.embedder = EmbeddingService.get_instance()
        self.vector_store = VectorStoreService.get_instance()
        self.reranker = RerankerService.get_instance()
        self.tag_retrieval = TagRetrievalService(self.vector_store, self.repo, self.reranker)

    async def create_tag(
        self,
        name: str,
        description: Optional[str] = None,
        force_create: bool = False,
        source: str = "user",
    ) -> TagCreateResult:
        existing = await self.repo.get_by_name(name)
        if existing:
            return TagCreateResult(
                created=False,
                tag=TagResponse.model_validate(existing),
                similar_tags=[],
                message="Tag already exists",
            )

        embedding = self.embedder.embed_one(name)

        if not force_create:
            similar = await self.tag_retrieval.retrieve(name, embedding)
            if similar:
                top_tag, top_score = similar[0]
                if top_score >= settings.TAG_DEDUP_THRESHOLD:
                    return TagCreateResult(
                        created=False,
                        tag=TagResponse.model_validate(top_tag),
                        similar_tags=[],
                        message="Similar tag exists",
                    )

        new_tag = await self.repo.create(name=name, description=description, source=source)

        qdrant_id = await self.vector_store.upsert_tag(new_tag.id, new_tag.name, embedding)
        await self.repo.update_qdrant_id(new_tag.id, qdrant_id)

        await self.db.commit()

        return TagCreateResult(
            created=True,
            tag=TagResponse.model_validate(new_tag),
            similar_tags=[],
            message="Created",
        )

    async def get_tag(self, tag_id: str):
        return await self.repo.get_by_id(tag_id)

    async def get_similar_tags(self, query: str, top_k: int = 5) -> List[SimilarTag]:
        embedding = self.embedder.embed_one(query)
        hits = await self.vector_store.search_similar_tags(embedding, top_k=top_k)
        if not hits:
            return []

        tag_ids = [h[0] for h in hits]
        vector_scores = {h[0]: h[1] for h in hits}
        db_tags = await self.repo.get_by_ids(tag_ids)
        if not db_tags:
            return []

        reranked = self.reranker.rerank(query, [t.name for t in db_tags])
        name_to_tag = {t.name: t for t in db_tags}

        results = []
        for name, rr_score in reranked[:top_k]:
            tag = name_to_tag.get(name)
            if not tag:
                continue
            v_score = vector_scores.get(tag.id, 0.0)
            results.append(SimilarTag(
                tag=TagResponse.model_validate(tag),
                score=round(0.7 * rr_score + 0.3 * v_score, 4),
                vector_score=round(v_score, 4),
                reranker_score=round(rr_score, 4),
            ))

        return results

    async def list_tags(self, limit: int = 100, offset: int = 0):
        return await self.repo.get_all(limit=limit, offset=offset)

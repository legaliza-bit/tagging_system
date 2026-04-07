import json
import logging
from typing import List, Optional

from app.config import settings
from app.db.repository import DocumentRepository, PendingReviewRepository
from app.services.infrastructure.embedding import EmbeddingService
from app.services.infrastructure.reranker import RerankerService
from app.services.infrastructure.retrieval import TagRetrievalService
from app.services.infrastructure.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class DocumentService:

    def __init__(self, db):
        self.doc_repo = DocumentRepository(db)
        self.pending_repo = PendingReviewRepository(db)
        self.db = db
        self.embedder = EmbeddingService.get_instance()
        self.vector_store = VectorStoreService.get_instance()
        self.reranker = RerankerService.get_instance()
        self.tag_retrieval = TagRetrievalService(
            self.vector_store, self.doc_repo, self.reranker
        )

    async def ingest(self, content: str, title: Optional[str] = None, dbpedia_label: Optional[str] = None):
        doc = await self.doc_repo.create(content=content, title=title, dbpedia_label=dbpedia_label)
        embedding = self.embedder.embed_one(content[:500])
        await self.vector_store.upsert_document(doc.id, embedding, content_snippet=content[:200])
        result = await self._tag(doc.id, content, embedding)
        await self.db.commit()
        return result

    async def get_documents_by_tag(self, tag_id: str, limit: int = 50) -> List:
        return await self.doc_repo.list_by_tag(tag_id, limit=limit)

    async def _tag(self, doc_id: str, text: str, embedding: List[float]) -> dict:
        scored = await self.tag_retrieval.retrieve(text, embedding)
        if not scored:
            return {"doc_id": doc_id, "tags": [], "uncertain": True}

        best_score = scored[0][1]

        if best_score < settings.UNCERTAINTY_THRESHOLD:
            candidates = [{"tag_id": t.id, "score": s} for t, s in scored[:5]]
            await self.pending_repo.create(
                document_id=doc_id,
                candidate_tags=json.dumps(candidates),
                max_confidence=best_score,
            )
            return {"doc_id": doc_id, "uncertain": True, "candidates": candidates}

        assignments = [
            {"tag_id": t.id, "confidence": s}
            for t, s in scored
            if s >= settings.TAG_ASSIGNMENT_THRESHOLD
        ] or [{"tag_id": scored[0][0].id, "confidence": scored[0][1]}]

        await self.doc_repo.assign_tags(doc_id, assignments)

        return {"doc_id": doc_id, "tags": assignments, "uncertain": False}

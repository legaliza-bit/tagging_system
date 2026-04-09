import logging
from typing import List, Optional

from app.config import settings
from app.db.repository import DocumentRepository, TagRepository
from app.services.infrastructure.embedding import EmbeddingService
from app.services.infrastructure.reranker import RerankerService
from app.services.infrastructure.retrieval import TagRetrievalService
from app.services.infrastructure.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class DocumentService:

    def __init__(self, db):
        self.doc_repo = DocumentRepository(db)
        self.tag_repo = TagRepository(db)
        self.db = db
        self.embedder = EmbeddingService.get_instance()
        self.vector_store = VectorStoreService.get_instance()
        self.reranker = RerankerService.get_instance()
        self.tag_retrieval = TagRetrievalService(
            self.vector_store, self.tag_repo, self.reranker
        )

    async def ingest(self, content: str, title: Optional[str] = None, dbpedia_label: Optional[str] = None):
        doc = await self.doc_repo.create(content=content, title=title, dbpedia_label=dbpedia_label)
        embedding = self.embedder.embed_one(content[:500])
        await self.vector_store.upsert_document(doc.id, embedding, content_snippet=content[:200])
        result = await self._tag(doc.id, content, embedding)
        await self.db.commit()
        return result

    async def retag_with_new_tag(self, tag_id: str, tag_name: str) -> int:
        """Run the full retrieval pipeline (vector search + reranker + relative
        scoring) for every document and assign the new tag only if it survives
        the margin filter and meets the assignment threshold."""
        tagged, offset, batch = 0, 0, 100

        while True:
            docs = await self.doc_repo.get_page(offset, batch)
            if not docs:
                break
            for doc in docs:
                embedding = self.embedder.embed_one(doc.content[:500])
                scored = await self.tag_retrieval.retrieve(doc.content, embedding)
                best_score = scored[0][1] if scored else 0
                tag_score = next((s for t, s in scored if t.id == tag_id), None)
                if tag_score is None:
                    logger.debug(f"Doc {doc.id}: '{tag_name}' not in top-K candidates")
                else:
                    logger.debug(
                        f"Doc {doc.id}: '{tag_name}' score={tag_score:.3f} "
                        f"best={best_score:.3f} threshold={settings.TAG_ASSIGNMENT_THRESHOLD}"
                    )
                for tag, score in scored:
                    if (
                        tag.id == tag_id
                        and score >= settings.TAG_ASSIGNMENT_THRESHOLD
                        and score >= best_score - settings.TAG_SCORE_MARGIN
                    ):
                        await self.doc_repo.assign_tags(
                            doc.id, [{"tag_id": tag_id, "confidence": score}]
                        )
                        tagged += 1
                        break
            await self.db.commit()
            offset += batch

        logger.info(f"Retroactive tagging '{tag_name}': {tagged} documents assigned.")
        return tagged

    async def get_documents_by_tag(self, tag_id: str, limit: int = 50) -> List:
        return await self.doc_repo.list_by_tag(tag_id, limit=limit)

    async def _tag(self, doc_id: str, text: str, embedding: List[float]) -> dict:
        scored = await self.tag_retrieval.retrieve(text, embedding)
        if not scored:
            return {"doc_id": doc_id, "tags": [], "uncertain": False}

        best_score = scored[0][1]
        assignments = [
            {"tag_id": t.id, "confidence": s}
            for t, s in scored
            if s >= settings.TAG_ASSIGNMENT_THRESHOLD
            and s >= best_score - settings.TAG_SCORE_MARGIN
        ] or [{"tag_id": scored[0][0].id, "confidence": scored[0][1]}]

        await self.doc_repo.assign_tags(doc_id, assignments)

        return {"doc_id": doc_id, "tags": assignments, "uncertain": False}

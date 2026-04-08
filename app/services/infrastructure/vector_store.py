import logging
import uuid
from typing import List, Optional, Tuple

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings, logger


class VectorStoreService:
    """Qdrant vector search for tags and documents."""

    _instance = None

    def __init__(self):
        self.client = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )
        self.tag_collection = settings.QDRANT_TAG_COLLECTION
        self.doc_collection = settings.QDRANT_DOC_COLLECTION
        self.dim = settings.EMBEDDING_DIM

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        cls._instance = None

    async def ensure_collections(self) -> None:
        existing = await self.client.get_collections()
        existing_names = [c.name for c in existing.collections]

        for name in (self.tag_collection, self.doc_collection):
            if name not in existing_names:
                await self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
                )
                logger.info(f"Created Qdrant collection: {name}")

    async def upsert_tag(self, tag_id: str, tag_name: str, embedding: List[float]) -> str:
        """Store tag embedding. Returns the Qdrant point ID."""
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, tag_id))
        await self.client.upsert(
            collection_name=self.tag_collection,
            points=[PointStruct(
                id=point_id,
                vector=embedding,
                payload={"tag_id": tag_id, "tag_name": tag_name},
            )],
        )
        return point_id

    async def upsert_document(
        self, doc_id: str, embedding: List[float], content_snippet: str = ""
    ) -> str:
        """Store document embedding. Returns the Qdrant point ID."""
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))
        await self.client.upsert(
            collection_name=self.doc_collection,
            points=[PointStruct(
                id=point_id,
                vector=embedding,
                payload={"doc_id": doc_id, "snippet": content_snippet[:200]},
            )],
        )
        return point_id

    async def search_similar_tags(
        self, embedding: List[float], top_k: int = 10, exclude_ids: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """ANN search in tag collection. Returns [(tag_id, score), ...] descending."""
        results = await self.client.query_points(
            collection_name=self.tag_collection,
            query=embedding,
            limit=top_k + (len(exclude_ids) if exclude_ids else 0),
            with_payload=True,
        )
        output = []
        for r in results.points:
            tag_id = r.payload.get("tag_id")
            if exclude_ids and tag_id in exclude_ids:
                continue
            output.append((tag_id, r.score))
        return output[:top_k]

    async def search_similar_documents(
        self, embedding: List[float], top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """ANN search in document collection. Returns [(doc_id, score), ...] descending."""
        results = await self.client.query_points(
            collection_name=self.doc_collection,
            query=embedding,
            limit=top_k,
            with_payload=True,
        )
        return [(r.payload.get("doc_id"), r.score) for r in results.points]

    async def delete_tag(self, point_id: str) -> None:
        await self.client.delete(
            collection_name=self.tag_collection,
            points_selector=[point_id],
        )

    async def collection_count(self, collection: str) -> int:
        info = await self.client.get_collection(collection)
        return info.points_count or 0

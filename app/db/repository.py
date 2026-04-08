from typing import Generic, List, Optional, TypeVar

from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.schemas import Document, DocumentTag, Tag

T = TypeVar("T")


class BaseRepository(Generic[T]):
    def __init__(self, db: AsyncSession, model):
        self.db = db
        self.model = model

    async def get_by_id(self, id_: str) -> Optional[T]:
        return await self.db.get(self.model, id_)

    async def get_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        result = await self.db.execute(
            select(self.model).offset(offset).limit(limit)
        )
        return result.scalars().all()

    async def get_by_ids(self, ids: List[str]) -> List[T]:
        if not ids:
            return []
        result = await self.db.execute(
            select(self.model).where(self.model.id.in_(ids))
        )
        return result.scalars().all()

    async def count(self) -> int:
        result = await self.db.execute(
            select(func.count()).select_from(self.model)
        )
        return result.scalar_one()

    async def create(self, **kwargs) -> T:
        obj = self.model(**kwargs)
        self.db.add(obj)
        await self.db.flush()
        return obj


class TagRepository(BaseRepository[Tag]):
    def __init__(self, db: AsyncSession):
        super().__init__(db, Tag)

    async def get_by_name(self, name: str) -> Optional[Tag]:
        result = await self.db.execute(
            select(Tag).where(Tag.name == name)
        )
        return result.scalar_one_or_none()

    async def update_qdrant_id(self, tag_id: str, qdrant_id: str) -> None:
        await self.db.execute(
            update(Tag).where(Tag.id == tag_id).values(qdrant_id=qdrant_id)
        )
        await self.db.flush()

    async def get_tag_doc_counts(self) -> dict:
        """Return {tag_id: document_count} for all tags that have assignments."""
        result = await self.db.execute(
            select(DocumentTag.tag_id, func.count(DocumentTag.document_id))
            .group_by(DocumentTag.tag_id)
        )
        return {tag_id: count for tag_id, count in result.all()}


class DocumentRepository(BaseRepository[Document]):
    def __init__(self, db: AsyncSession):
        super().__init__(db, Document)

    async def get_all(self, limit: int = 100, offset: int = 0) -> List[Document]:
        result = await self.db.execute(
            select(Document)
            .options(selectinload(Document.document_tags).selectinload(DocumentTag.tag))
            .offset(offset).limit(limit)
        )
        return result.scalars().all()

    async def get_by_id(self, id_: str) -> Optional[Document]:
        result = await self.db.execute(
            select(Document)
            .options(selectinload(Document.document_tags).selectinload(DocumentTag.tag))
            .where(Document.id == id_)
        )
        return result.scalar_one_or_none()

    async def get_page(self, offset: int, limit: int) -> List[Document]:
        """Fetch a page of documents without eager-loading relations."""
        result = await self.db.execute(
            select(Document).offset(offset).limit(limit)
        )
        return result.scalars().all()

    async def list_by_tag(self, tag_id: str, limit: int = 50) -> List[Document]:
        result = await self.db.execute(
            select(Document)
            .options(selectinload(Document.document_tags).selectinload(DocumentTag.tag))
            .join(DocumentTag, Document.id == DocumentTag.document_id)
            .where(DocumentTag.tag_id == tag_id)
            .limit(limit)
        )
        return result.scalars().all()

    async def assign_tags(self, doc_id: str, assignments: List[dict]) -> None:
        """Persist tag assignments and update document tagging status.

        assignments: [{"tag_id": ..., "confidence": ...}, ...]
        """
        for a in assignments:
            dt_obj = DocumentTag(
                document_id=doc_id,
                tag_id=a["tag_id"],
                confidence=a["confidence"],
            )
            await self.db.merge(dt_obj)

        max_conf = max(a["confidence"] for a in assignments)
        await self.db.execute(
            update(Document)
            .where(Document.id == doc_id)
            .values(is_tagged=True, tagging_confidence=max_conf)
        )
        await self.db.flush()

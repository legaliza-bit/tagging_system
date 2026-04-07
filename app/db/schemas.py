import uuid
import datetime as dt
from sqlalchemy import (
    Column, String, Text, Float, DateTime, ForeignKey,
    Boolean, Integer, Table
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, DeclarativeBase
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.config import settings


def gen_uuid():
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class DocumentTag(Base):
    __tablename__ = "document_tags"

    document_id = Column(UUID(as_uuid=False), ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True)
    tag_id = Column(UUID(as_uuid=False), ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True)

    confidence = Column(Float)
    is_human_verified = Column(Boolean, default=False)
    assigned_at = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc))

    document = relationship("Document", back_populates="document_tags")
    tag = relationship("Tag", back_populates="document_tags")


class Tag(Base):
    __tablename__ = "tags"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    name = Column(String(256), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    qdrant_id = Column(String(64), nullable=True, unique=True)  # point ID in Qdrant
    created_at = Column(DateTime, default=dt.datetime.now(dt.timezone.utc))
    updated_at = Column(DateTime, default=dt.datetime.now(dt.timezone.utc), onupdate=dt.datetime.now(dt.timezone.utc))
    is_active = Column(Boolean, default=True)
    source = Column(String(64), default="user")
    document_tags = relationship("DocumentTag", back_populates="tag")


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    title = Column(String(512), nullable=True)
    content = Column(Text, nullable=False)
    qdrant_id = Column(String(64), nullable=True, unique=True)
    created_at = Column(DateTime, default=dt.datetime.now(dt.timezone.utc))
    updated_at = Column(DateTime, default=dt.datetime.now(dt.timezone.utc), onupdate=dt.datetime.now(dt.timezone.utc))
    is_tagged = Column(Boolean, default=False)
    tagging_confidence = Column(Float, nullable=True)  # overall max confidence
    dbpedia_label = Column(String(256), nullable=True)   # ground truth for eval
    document_tags = relationship("DocumentTag", back_populates="document")
    tags = relationship("Tag", secondary="document_tags", viewonly=True)


class PendingReview(Base):
    """Stores low-confidence predictions awaiting human review."""
    __tablename__ = "pending_reviews"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    document_id = Column(UUID(as_uuid=False), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    candidate_tags = Column(Text, nullable=False)
    max_confidence = Column(Float, nullable=False)
    resolved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=dt.datetime.now(dt.timezone.utc))
    resolved_at = Column(DateTime, nullable=True)

    document = relationship("Document", lazy="selectin")


engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

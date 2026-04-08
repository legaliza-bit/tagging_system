from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.schemas import AsyncSessionLocal, get_db
from app.schemas.tag import SimilarTag, TagCreate, TagCreateResult, TagResponse
from app.services.application.document_service import DocumentService
from app.services.application.tag_service import TagService

router = APIRouter(prefix="/tags", tags=["tags"])


async def _retag_documents(tag_id: str, tag_name: str) -> None:
    async with AsyncSessionLocal() as db:
        svc = DocumentService(db)
        await svc.retag_with_new_tag(tag_id, tag_name)


@router.post("/", response_model=TagCreateResult, status_code=201)
async def create_tag(
    payload: TagCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Create a tag with semantic deduplication.
    Use force_create=True to bypass."""
    svc = TagService(db)
    result = await svc.create_tag(
        name=payload.name,
        description=payload.description,
        force_create=payload.force_create,
    )
    if result.created:
        background_tasks.add_task(
            _retag_documents, result.tag.id, result.tag.name
        )
    return result


@router.get("/search", response_model=List[SimilarTag])
async def search_similar_tags(
    q: str = Query(..., min_length=2),
    top_k: int = Query(5, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
):
    """Semantic tag search: bi-encoder recall + cross-encoder reranking."""
    svc = TagService(db)
    return await svc.get_similar_tags(q, top_k=top_k)


@router.get("/", response_model=List[TagResponse])
async def list_tags(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    svc = TagService(db)
    tags = await svc.list_tags(limit=limit, offset=offset)
    return [TagResponse.model_validate(t) for t in tags]


@router.get("/{tag_id}", response_model=TagResponse)
async def get_tag(tag_id: str, db: AsyncSession = Depends(get_db)):
    svc = TagService(db)
    tag = await svc.get_tag(tag_id)
    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")
    return TagResponse.model_validate(tag)

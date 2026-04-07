from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from tagging_service.app.db.schemas import get_db
from app.schemas.document import DocumentCreate, DocumentResponse
from app.services.application.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/", response_model=dict, status_code=201)
async def ingest_document(
    payload: DocumentCreate, db: AsyncSession = Depends(get_db)
):
    """Ingest a document and auto-tag it."""
    svc = DocumentService(db)
    return await svc.ingest(
        content=payload.content,
        title=payload.title,
        dbpedia_label=payload.dbpedia_label,
    )


@router.get("/", response_model=List[DocumentResponse])
async def list_documents(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    svc = DocumentService(db)
    docs = await svc.doc_repo.get_all(limit=limit, offset=offset)
    return [DocumentResponse.model_validate(d) for d in docs]


@router.get("/by-tag/{tag_id}", response_model=List[DocumentResponse])
async def documents_by_tag(
    tag_id: str,
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Retrieve all documents associated with a specific tag."""
    svc = DocumentService(db)
    docs = await svc.get_documents_by_tag(tag_id, limit=limit)
    return [DocumentResponse.model_validate(d) for d in docs]


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str, db: AsyncSession = Depends(get_db)):
    svc = DocumentService(db)
    doc = await svc.doc_repo.get_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse.model_validate(doc)

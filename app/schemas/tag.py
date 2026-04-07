from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class TagCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=256)
    description: Optional[str] = None
    force_create: bool = False


class TagResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    source: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class SimilarTag(BaseModel):
    tag: TagResponse
    score: float
    vector_score: float
    reranker_score: float


class TagCreateResult(BaseModel):
    created: bool
    tag: TagResponse
    similar_tags: List[SimilarTag] = []
    message: str


class TagWithDocCount(BaseModel):
    id: str
    name: str
    description: Optional[str]
    source: str
    doc_count: int

    model_config = {"from_attributes": True}

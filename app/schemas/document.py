from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class TagBrief(BaseModel):
    id: str
    name: str
    confidence: Optional[float] = None
    is_human_verified: bool = False

    model_config = {"from_attributes": True}


class DocumentCreate(BaseModel):
    content: str = Field(..., min_length=10, max_length=10000)
    title: Optional[str] = Field(None, max_length=512)
    dbpedia_label: Optional[str] = None


class DocumentResponse(BaseModel):
    id: str
    title: Optional[str]
    content: str
    is_tagged: bool
    tagging_confidence: Optional[float]
    dbpedia_label: Optional[str]
    tags: List[TagBrief] = []
    created_at: datetime

    model_config = {"from_attributes": True}


class TaggingResult(BaseModel):
    document_id: str
    assigned_tags: List[TagBrief]
    uncertain: bool
    candidates: List[TagBrief] = []
    message: str

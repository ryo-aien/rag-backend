from datetime import datetime

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    k: int = Field(default=4, ge=1, le=20)
    metadata_filter: dict | None = None


class IndexRequest(BaseModel):
    directory: str | None = None


class IndexResponse(BaseModel):
    status: str
    message: str


class UploadResponse(BaseModel):
    status: str
    filename: str
    message: str


class DocumentInfo(BaseModel):
    filename: str
    size_bytes: int
    updated_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]

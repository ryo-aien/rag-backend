import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
from sse_starlette.sse import EventSourceResponse

from app.core.config import settings
from app.core.indexing import LOADER_MAP, delete_document, run_indexing
from app.core.rag import generate_answer_stream
from app.models.schemas import (
    DeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    IndexRequest,
    IndexResponse,
    QueryRequest,
    UploadResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["v1"])


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    data_dir = Path(settings.data_dir)
    if not data_dir.exists():
        return DocumentListResponse(documents=[])

    documents = []
    for file_path in sorted(data_dir.iterdir()):
        if file_path.is_dir() or file_path.suffix.lower() not in LOADER_MAP:
            continue
        stat = file_path.stat()
        documents.append(
            DocumentInfo(
                filename=file_path.name,
                size_bytes=stat.st_size,
                updated_at=datetime.fromtimestamp(stat.st_mtime),
                file_type=file_path.suffix.lower(),
            )
        )
    return DocumentListResponse(documents=documents)


@router.post("/query")
async def query(request: QueryRequest):
    async def event_generator():
        async for token in generate_answer_stream(
            question=request.question,
            k=request.k,
            metadata_filter=request.metadata_filter,
        ):
            yield {"data": token}

    return EventSourceResponse(event_generator())


@router.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_indexing, request.directory)
    return IndexResponse(
        status="accepted",
        message="Indexing started in background",
    )


@router.delete("/documents/{filename}", response_model=DeleteResponse)
async def delete_document_endpoint(filename: str):
    data_dir = Path(settings.data_dir)
    file_path = data_dir / filename

    # パストラバーサル防止
    if file_path.resolve().parent != data_dir.resolve():
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    try:
        result = delete_document(filename)
    except Exception as e:
        logger.exception("Failed to delete document: %s", filename)
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}") from e

    return DeleteResponse(
        status="success",
        filename=filename,
        deleted_vectors=result["deleted_vectors"],
        deleted_records=result["deleted_records"],
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile, background_tasks: BackgroundTasks):
    try:
        data_dir = Path(settings.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        file_path = data_dir / file.filename
        content = await file.read()
        file_path.write_bytes(content)

        logger.info("File uploaded: %s", file.filename)
    except Exception as e:
        logger.exception("File upload failed")
        raise HTTPException(status_code=500, detail=f"File upload failed: {e}") from e

    background_tasks.add_task(run_indexing)
    return UploadResponse(
        status="success",
        filename=file.filename,
        message="File uploaded and indexing started",
    )

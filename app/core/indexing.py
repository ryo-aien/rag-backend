import logging
from collections.abc import Iterator
from pathlib import Path

import chromadb
from langchain.indexes import SQLRecordManager, index
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings

logger = logging.getLogger(__name__)

LOADER_MAP: dict[str, type] = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".csv": CSVLoader,
    ".md": UnstructuredMarkdownLoader,
}

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(openai_api_key=settings.openai_api_key)


def _get_vectorstore() -> Chroma:
    client = chromadb.HttpClient(host=settings.chroma_host, port=int(settings.chroma_port))
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=_get_embeddings(),
        client=client,
    )


def _get_record_manager() -> SQLRecordManager:
    return SQLRecordManager(
        namespace=settings.record_manager_namespace,
        db_url=settings.postgres_url,
    )


def load_documents_lazy(directory: str | None = None) -> Iterator[Document]:
    data_dir = Path(directory or settings.data_dir)
    if not data_dir.exists():
        logger.warning("Data directory does not exist: %s", data_dir)
        return

    for file_path in sorted(data_dir.iterdir()):
        if file_path.is_dir() or file_path.suffix.lower() not in LOADER_MAP:
            continue

        try:
            loader_cls = LOADER_MAP[file_path.suffix.lower()]
            loader = loader_cls(str(file_path))
            docs = loader.load()
        except Exception:
            logger.exception("Failed to load file: %s", file_path)
            continue

        chunks = text_splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata["source"] = file_path.name
            chunk.metadata.setdefault("page", 0)
            yield chunk


def run_indexing(directory: str | None = None) -> dict:
    vectorstore = _get_vectorstore()
    record_manager = _get_record_manager()

    batch: list[Document] = []
    total_indexed = 0
    errors = 0

    for doc in load_documents_lazy(directory):
        batch.append(doc)
        if len(batch) >= 100:
            try:
                result = index(batch, record_manager, vectorstore, cleanup="incremental", source_id_key="source")
                total_indexed += result.get("num_added", 0) + result.get("num_updated", 0)
                logger.info("Batch indexed: %s", result)
            except Exception:
                errors += 1
                logger.exception("Batch indexing failed, skipping batch")
            batch = []

    if batch:
        try:
            result = index(batch, record_manager, vectorstore, cleanup="incremental", source_id_key="source")
            total_indexed += result.get("num_added", 0) + result.get("num_updated", 0)
            logger.info("Final batch indexed: %s", result)
        except Exception:
            errors += 1
            logger.exception("Final batch indexing failed")

    return {"total_indexed": total_indexed, "errors": errors}


def ensure_record_manager_schema() -> None:
    record_manager = _get_record_manager()
    record_manager.create_schema()
    logger.info("Record manager schema ensured")

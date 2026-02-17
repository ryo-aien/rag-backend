import json
import logging
from collections.abc import Iterator
from datetime import datetime
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
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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


def _get_embeddings() -> Embeddings:
    """EMBEDDING_PROVIDER設定に応じたOpenAIEmbeddingsインスタンスを返す。"""
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddings(openai_api_key=settings.openai_api_key)
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=settings.openai_api_key,
    )


def _get_vectorstore() -> Chroma:
    """ChromaDBにHTTP接続し、ベクトルストアインスタンスを返す。"""
    client = chromadb.HttpClient(host=settings.chroma_host, port=int(settings.chroma_port))
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=_get_embeddings(),
        client=client,
    )


def _get_record_manager() -> SQLRecordManager:
    """PostgreSQLベースのレコードマネージャーを返す。インデックス済みドキュメントの追跡に使用。"""
    return SQLRecordManager(
        namespace=settings.record_manager_namespace,
        db_url=settings.postgres_url,
    )


def _infer_metadata(text: str) -> dict:
    """ファイルの先頭テキストをGPT-4oに渡し、categoryとdepartmentを推測する。"""
    truncated = text[:2000]
    prompt = (
        "以下の社内ドキュメントの内容から、カテゴリと所管部署を推測してJSON形式で返してください。\n\n"
        "カテゴリ候補: 規程, マニュアル, ガイドライン, FAQ, 報告書, 議事録, お知らせ, その他\n"
        "所管部署が不明な場合は「全社共通」としてください。\n\n"
        '{"category": "...", "department": "..."}\n\n'
        f"ドキュメント内容:\n{truncated}"
    )
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=settings.openai_api_key,
        )
        response = llm.invoke([{"role": "user", "content": prompt}])
        content = response.content.strip()
        # JSON部分を抽出（コードブロックで囲まれている場合に対応）
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        result = json.loads(content)
        return {
            "category": result.get("category", "その他"),
            "department": result.get("department", "全社共通"),
        }
    except Exception:
        logger.exception("Failed to infer metadata via LLM")
        return {"category": "その他", "department": "全社共通"}


def load_documents_lazy(directory: str | None = None) -> Iterator[Document]:
    """指定ディレクトリのファイルを読み込み、チャンク分割したDocumentを逐次返す。"""
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

        file_type = file_path.suffix.lower()
        created_at = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()

        full_text = "\n".join(doc.page_content for doc in docs)
        inferred = _infer_metadata(full_text)

        chunks = text_splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata["source"] = file_path.name
            chunk.metadata.setdefault("page", 0)
            chunk.metadata["file_type"] = file_type
            chunk.metadata["created_at"] = created_at
            chunk.metadata["category"] = inferred["category"]
            chunk.metadata["department"] = inferred["department"]
            yield chunk


def run_indexing(directory: str | None = None) -> dict:
    """ドキュメントを100件ずつバッチでChromaDBにインクリメンタルインデックスする。"""
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


def delete_document(filename: str) -> dict:
    """指定ファイルに関連するデータをChromaDBとレコードマネージャーから削除する。"""
    vectorstore = _get_vectorstore()
    record_manager = _get_record_manager()

    # ChromaDBからsourceが一致するドキュメントを検索して削除
    collection = vectorstore._collection
    results = collection.get(where={"source": {"$eq": filename}})
    deleted_vectors = len(results["ids"]) if results["ids"] else 0
    if results["ids"]:
        collection.delete(ids=results["ids"])
    logger.info("Deleted %d vectors from ChromaDB for: %s", deleted_vectors, filename)

    # レコードマネージャーからsourceが一致するレコードを削除
    deleted_records = 0
    try:
        keys = record_manager.list_keys(group_ids=[filename])
        if keys:
            deleted_records = len(keys)
            record_manager.delete_keys(keys)
        logger.info("Deleted %d records from record manager for: %s", deleted_records, filename)
    except Exception:
        logger.exception("Failed to delete records from record manager for: %s", filename)

    # ファイルシステムからファイルを削除
    file_path = Path(settings.data_dir) / filename
    if file_path.exists():
        file_path.unlink()
        logger.info("Deleted file: %s", file_path)

    return {"deleted_vectors": deleted_vectors, "deleted_records": deleted_records}


def ensure_record_manager_schema() -> None:
    """レコードマネージャーのDBスキーマが存在しなければ作成する。"""
    record_manager = _get_record_manager()
    record_manager.create_schema()
    logger.info("Record manager schema ensured")

import logging
from collections.abc import AsyncIterator

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
あなたはRAGアシスタントです。ドキュメントに基づいて質問に回答するアシスタントです。
以下のコンテキスト情報を使用して、ユーザーの質問に正確に回答してください。
コンテキストに情報がない場合は、「その情報は見つかりませんでした」と回答してください。

コンテキスト:
{context}
"""


def _get_vectorstore() -> Chroma:
    client = chromadb.HttpClient(host=settings.chroma_host, port=int(settings.chroma_port))
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=OpenAIEmbeddings(openai_api_key=settings.openai_api_key),
        client=client,
    )


def build_chroma_filter(metadata_filter: dict | None) -> dict | None:
    if not metadata_filter:
        return None

    conditions = [{k: {"$eq": v}} for k, v in metadata_filter.items()]

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}


def retrieve_documents(question: str, k: int = 4, metadata_filter: dict | None = None) -> list[Document]:
    vectorstore = _get_vectorstore()
    where = build_chroma_filter(metadata_filter)

    kwargs: dict = {"k": k}
    if where:
        kwargs["filter"] = where

    return vectorstore.similarity_search(question, **kwargs)


def _format_context(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 0)
        parts.append(f"[ソース: {source}, ページ: {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


async def generate_answer_stream(
    question: str,
    k: int = 4,
    metadata_filter: dict | None = None,
) -> AsyncIterator[str]:
    try:
        docs = retrieve_documents(question, k, metadata_filter)
        context = _format_context(docs)

        llm = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=settings.openai_api_key,
            streaming=True,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": question},
        ]

        async for chunk in llm.astream(messages):
            if chunk.content:
                yield chunk.content

    except Exception:
        logger.exception("Error during RAG generation")
        yield "[ERROR] 回答の生成中にエラーが発生しました。"

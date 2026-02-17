import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    postgres_user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "rag"))
    postgres_password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "rag_password"))
    postgres_db: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "rag_records"))
    postgres_host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "postgres"))
    postgres_port: str = field(default_factory=lambda: os.getenv("POSTGRES_PORT", "5432"))

    chroma_host: str = field(default_factory=lambda: os.getenv("CHROMA_HOST", "chroma"))
    chroma_port: str = field(default_factory=lambda: os.getenv("CHROMA_PORT", "8000"))
    chroma_collection: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "rag_documents"))

    data_dir: str = field(default_factory=lambda: os.getenv("DATA_DIR", "./data"))

    # エンベディングプロバイダー: "open" (text-embedding-3-small) または "openai" (OpenAIEmbeddingsデフォルト)
    embedding_provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "open"))

    record_manager_namespace: str = "chroma/rag_documents"

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def chroma_url(self) -> str:
        return f"http://{self.chroma_host}:{self.chroma_port}"


settings = Settings()

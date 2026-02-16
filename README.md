# RAG API Server

ドキュメントをアップロード・インデックスし、自然言語で質問応答できる RAG (Retrieval-Augmented Generation) API サーバーです。

## 技術スタック

- **API**: FastAPI + Uvicorn
- **LLM / Embedding**: OpenAI (GPT-4o / text-embedding)
- **ベクトルDB**: ChromaDB
- **レコード管理**: PostgreSQL + LangChain SQLRecordManager
- **コンテナ**: Docker Compose

## アーキテクチャ

```
クライアント → FastAPI (/v1) → LangChain → OpenAI API
                                    ↓
                              ChromaDB (ベクトル検索)
                              PostgreSQL (レコード管理)
```

## セットアップ

### 1. 環境変数の設定

`.env` ファイルをプロジェクトルートに作成します。

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
POSTGRES_USER=rag
POSTGRES_PASSWORD=rag_password
POSTGRES_DB=rag_records
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
CHROMA_HOST=chroma
CHROMA_PORT=8000
CHROMA_COLLECTION=rag_documents
DATA_DIR=./data
```

### 2. 起動

```bash
docker compose up --build -d
```

API は `http://localhost:8000` で起動します。

## API エンドポイント

### ヘルスチェック

```
GET /health
```

### 質問応答 (SSE ストリーミング)

```
POST /v1/query
Content-Type: application/json

{
  "question": "〇〇について教えてください",
  "k": 4,
  "metadata_filter": {"source": "example.pdf"}
}
```

- `question` (必須): 質問文
- `k` (任意, デフォルト: 4): 検索する関連ドキュメント数 (1〜20)
- `metadata_filter` (任意): メタデータによるフィルタリング

レスポンスは Server-Sent Events (SSE) 形式でストリーミング返却されます。

### ファイルアップロード

```
POST /v1/upload
Content-Type: multipart/form-data

file: <アップロードするファイル>
```

ファイルを `data/` に保存し、バックグラウンドでインデックスを実行します。

### インデックス実行

```
POST /v1/index
Content-Type: application/json

{
  "directory": null
}
```

`data/` ディレクトリ内のドキュメントをバックグラウンドでインデックスします。

## 対応ファイル形式

| 拡張子 | ローダー |
|--------|----------|
| `.txt` | TextLoader |
| `.pdf` | PyPDFLoader |
| `.csv` | CSVLoader |
| `.md`  | UnstructuredMarkdownLoader |

## プロジェクト構成

```
app/
├── main.py            # FastAPI アプリケーション定義
├── core/
│   ├── config.py      # 環境変数・設定管理
│   ├── indexing.py     # ドキュメント読み込み・インデックス処理
│   └── rag.py         # 検索・回答生成 (RAG パイプライン)
├── models/
│   └── schemas.py     # リクエスト/レスポンススキーマ
└── routers/
    └── v1.py          # v1 API ルーター
```

# フロントエンド開発仕様書

## 1. 概要

RAG API サーバーに対応するフロントエンドアプリケーションの開発仕様。
ユーザーがドキュメントをアップロードし、自然言語で質問・回答を得られるチャット UI を提供する。

---

## 2. API インターフェース仕様

### ベース URL

```
http://localhost:8000
```

### 2.1 ヘルスチェック

| 項目 | 値 |
|---|---|
| メソッド | `GET` |
| パス | `/health` |
| Content-Type (レスポンス) | `application/json` |

**レスポンス**:
```json
{"status": "ok"}
```

---

### 2.2 質問応答 (SSE ストリーミング)

| 項目 | 値 |
|---|---|
| メソッド | `POST` |
| パス | `/v1/query` |
| Content-Type (リクエスト) | `application/json` |
| Content-Type (レスポンス) | `text/event-stream` |

**リクエストボディ**:

| フィールド | 型 | 必須 | デフォルト | バリデーション | 説明 |
|---|---|---|---|---|---|
| `question` | string | Yes | - | 空文字不可 | 質問文 |
| `k` | integer | No | 4 | 1 以上 20 以下 | 取得する関連ドキュメント数 |
| `metadata_filter` | object \| null | No | null | - | メタデータフィルタ (例: `{"source": "file.pdf"}`) |

**SSE イベント形式**:

```
event: message
data: <1トークン分のテキスト>

```

- 回答はトークン単位で順次送信される
- 全トークンを連結すると回答全文になる
- エラー時は `[ERROR] 回答の生成中にエラーが発生しました。` がデータとして送信される
- ストリーム終了時にコネクションが閉じられる

---

### 2.3 ファイルアップロード

| 項目 | 値 |
|---|---|
| メソッド | `POST` |
| パス | `/v1/upload` |
| Content-Type (リクエスト) | `multipart/form-data` |
| Content-Type (レスポンス) | `application/json` |

**リクエスト**:

| フィールド | 型 | 必須 | 説明 |
|---|---|---|---|
| `file` | File (バイナリ) | Yes | アップロードするファイル |

**成功レスポンス** (HTTP 200):
```json
{
  "status": "success",
  "filename": "example.pdf",
  "message": "File uploaded and indexing started"
}
```

**エラーレスポンス** (HTTP 500):
```json
{
  "detail": "File upload failed: <エラー詳細>"
}
```

**備考**:
- アップロード後、バックグラウンドでインデックスが自動実行される
- インデックス完了を通知する API は現状存在しない（Fire-and-Forget 方式）

---

### 2.4 インデックス実行

| 項目 | 値 |
|---|---|
| メソッド | `POST` |
| パス | `/v1/index` |
| Content-Type (リクエスト) | `application/json` |
| Content-Type (レスポンス) | `application/json` |

**リクエストボディ**:

| フィールド | 型 | 必須 | デフォルト | 説明 |
|---|---|---|---|---|
| `directory` | string \| null | No | null | 対象ディレクトリパス (null でサーバーデフォルトの `data/`) |

**レスポンス** (HTTP 200):
```json
{
  "status": "accepted",
  "message": "Indexing started in background"
}
```

---

### 2.5 ドキュメント一覧取得

| 項目 | 値 |
|---|---|
| メソッド | `GET` |
| パス | `/v1/documents` |
| Content-Type (レスポンス) | `application/json` |

**リクエスト**: パラメータなし

**成功レスポンス** (HTTP 200):
```json
{
  "documents": [
    {
      "filename": "会社概要.txt",
      "size_bytes": 2048,
      "updated_at": "2026-02-16T12:00:00"
    },
    {
      "filename": "製品カタログ.pdf",
      "size_bytes": 1048576,
      "updated_at": "2026-02-15T09:30:00"
    }
  ]
}
```

**レスポンスフィールド**:

| フィールド | 型 | 説明 |
|---|---|---|
| `documents` | array | ドキュメント情報の配列 |
| `documents[].filename` | string | ファイル名（拡張子付き） |
| `documents[].size_bytes` | integer | ファイルサイズ（バイト単位） |
| `documents[].updated_at` | string (ISO 8601) | 最終更新日時（例: `2026-02-16T12:00:00`） |

**備考**:
- `data/` ディレクトリに存在するファイルのうち、対応拡張子（`.txt`, `.pdf`, `.csv`, `.md`）のみが返却される
- ファイルが存在しない場合、`documents` は空配列 `[]` になる
- ファイル名のアルファベット順でソートされる
- アップロード直後に呼び出すことで、新しいファイルが一覧に含まれることを確認できる

**TypeScript 型定義**:
```typescript
interface DocumentInfo {
  filename: string;
  size_bytes: number;
  updated_at: string; // ISO 8601
}

interface DocumentListResponse {
  documents: DocumentInfo[];
}
```

**フロントエンド実装ガイド**:
```typescript
// API呼び出し例
const fetchDocuments = async (): Promise<DocumentListResponse> => {
  const res = await fetch("http://localhost:8000/v1/documents");
  if (!res.ok) throw new Error(`Failed to fetch documents: ${res.status}`);
  return res.json();
};
```

**想定ユースケース**:
- ページ読み込み時にドキュメント一覧を取得し、サイドバー等に表示する
- ファイルアップロード完了後に再取得して一覧を更新する
- `size_bytes` はフロントエンド側で KB/MB 等に変換して表示する

---

## 3. 対応ファイル形式

フロントエンドでアップロードを許可すべきファイル形式:

| 拡張子 | MIME タイプ | 内容 |
|---|---|---|
| `.txt` | `text/plain` | プレーンテキスト |
| `.pdf` | `application/pdf` | PDF ドキュメント |
| `.csv` | `text/csv` | CSV データ |
| `.md` | `text/markdown` | Markdown ドキュメント |

---

## 4. エラーレスポンス共通仕様

FastAPI のデフォルトエラー形式に準拠する。

**バリデーションエラー** (HTTP 422):
```json
{
  "detail": [
    {
      "loc": ["body", "question"],
      "msg": "Field required",
      "type": "missing"
    }
  ]
}
```

**サーバーエラー** (HTTP 500):
```json
{
  "detail": "<エラーメッセージ>"
}
```

---

## 5. CORS

バックエンドには `CORSMiddleware` が設定済み（全オリジン許可）。
フロントエンドを別オリジン（例: `http://localhost:3000`）で動作させても、追加設定なしで API 呼び出しが可能。

---

## 6. SSE 詳細仕様

バックエンドは `sse-starlette` ライブラリを使用。

| 項目 | 値 |
|---|---|
| イベント名 | `message` |
| データ形式 | プレーンテキスト (1トークン) |
| 区切り | 空行 (`\n\n`) |
| 終了条件 | コネクション切断 |
| エラー識別 | データが `[ERROR]` で始まる |

イベントストリーム例:
```
event: message
data: こんにちは

event: message
data: 、RAG

event: message
data: アシスタント

event: message
data: です。

```

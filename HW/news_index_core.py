"""
Shared logic to build the HW7 Chroma news index from CSV.
Used by scripts/build_news_index.py and HW7 Streamlit (auto-build on deploy).
"""
from __future__ import annotations

import os

import chromadb
from chromadb.utils import embedding_functions
import pandas as pd

# Project root: parent of HW/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CHROMA_PATH = os.path.join(_PROJECT_ROOT, "chroma_news_hw7")
COLLECTION_NAME = "NewsHW7"
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_CHUNK_CHARS = 1800
OVERLAP = 200


def project_root() -> str:
    return _PROJECT_ROOT


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = OVERLAP) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


def resolve_csv_path(explicit: str | None = None) -> str | None:
    candidates = []
    if explicit:
        candidates.append(os.path.abspath(explicit))
    candidates.extend(
        [
            os.path.join(_PROJECT_ROOT, "data", "news.csv"),
            os.path.join(_PROJECT_ROOT, "news.csv"),
            os.path.expanduser("~/Downloads/news.csv"),
        ]
    )
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def build_news_index_from_csv(api_key: str, csv_path: str):
    """
    (Re)build the NewsHW7 collection from scratch: delete if exists, create, embed all chunks.
    """
    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    required = {"company_name", "Date", "Document", "URL"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBEDDING_MODEL,
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
        metadata={"description": "HW7 news articles from CSV"},
    )

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for idx, row in df.iterrows():
        article_id = f"article_{idx}"
        company = str(row.get("company_name", "")).strip() or "Unknown"
        date = str(row.get("Date", "")).strip()
        url = str(row.get("URL", "")).strip()
        doc_text = str(row.get("Document", "")).strip()
        if not doc_text:
            continue
        chunks = chunk_text(doc_text)
        if not chunks:
            continue
        for ci, chunk in enumerate(chunks):
            chunk_id = f"{article_id}__c{ci}"
            documents.append(chunk)
            metadatas.append(
                {
                    "article_id": article_id,
                    "company_name": company[:500],
                    "date": date[:200],
                    "url": url[:2000],
                    "chunk_index": ci,
                }
            )
            ids.append(chunk_id)

    if not documents:
        raise ValueError("No documents to index from CSV.")

    batch = 100
    for i in range(0, len(documents), batch):
        collection.add(
            documents=documents[i : i + batch],
            metadatas=metadatas[i : i + batch],
            ids=ids[i : i + batch],
        )
        print(f"Added batch {i // batch + 1} ({min(i + batch, len(documents))}/{len(documents)} chunks)")

    print(f"Done. Collection {COLLECTION_NAME} has {collection.count()} chunks at {CHROMA_PATH}")
    return collection

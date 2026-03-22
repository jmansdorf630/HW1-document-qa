#!/usr/bin/env python3
"""
Build the ChromaDB index for HW7 news bot (run once outside Streamlit).

Usage:
  export OPENAI_API_KEY=sk-...
  python scripts/build_news_index.py [--csv PATH]

Looks for news.csv in: --csv, data/news.csv, project root news.csv, ~/Downloads/news.csv
"""
from __future__ import annotations

import argparse
import os
import sys

# Project root = parent of scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import chromadb
from chromadb.utils import embedding_functions
import pandas as pd

CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma_news_hw7")
COLLECTION_NAME = "NewsHW7"
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_CHUNK_CHARS = 1800
OVERLAP = 200


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


def resolve_csv_path(explicit: str | None) -> str | None:
    candidates = []
    if explicit:
        candidates.append(os.path.abspath(explicit))
    candidates.extend(
        [
            os.path.join(PROJECT_ROOT, "data", "news.csv"),
            os.path.join(PROJECT_ROOT, "news.csv"),
            os.path.expanduser("~/Downloads/news.csv"),
        ]
    )
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chroma news index for HW7")
    parser.add_argument("--csv", type=str, default=None, help="Path to news.csv")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection and rebuild",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY in the environment.", file=sys.stderr)
        sys.exit(1)

    csv_path = resolve_csv_path(args.csv)
    if not csv_path:
        print(
            "Could not find news.csv. Place it in data/news.csv or pass --csv PATH",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    required = {"company_name", "Date", "Document", "URL"}
    missing = required - set(df.columns)
    if missing:
        print(f"CSV missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBEDDING_MODEL,
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if args.reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted collection {COLLECTION_NAME}")
        except Exception:
            pass

    collection = None
    try:
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=openai_ef)
    except Exception:
        pass

    if collection is not None and collection.count() > 0 and not args.reset:
        print(
            f"Collection {COLLECTION_NAME} already has {collection.count()} chunks. "
            "Use --reset to rebuild."
        )
        return

    if args.reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection = None

    if collection is None:
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
        print("No documents to index.", file=sys.stderr)
        sys.exit(1)

    batch = 100
    for i in range(0, len(documents), batch):
        collection.add(
            documents=documents[i : i + batch],
            metadatas=metadatas[i : i + batch],
            ids=ids[i : i + batch],
        )
        print(f"Added batch {i // batch + 1} ({min(i + batch, len(documents))}/{len(documents)} chunks)")

    print(f"Done. Collection {COLLECTION_NAME} has {collection.count()} chunks at {CHROMA_PATH}")


if __name__ == "__main__":
    main()

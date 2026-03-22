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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import chromadb
from chromadb.utils import embedding_functions

from HW.news_index_core import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    build_news_index_from_csv,
    resolve_csv_path,
)


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

    try:
        build_news_index_from_csv(api_key, csv_path)
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

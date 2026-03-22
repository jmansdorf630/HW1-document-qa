"""
HW7 – News reporting bot: answers only from indexed CSV articles (RAG).
Pre-build the index with: python scripts/build_news_index.py
"""
from __future__ import annotations

import os
import re
import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# Same paths as scripts/build_news_index.py
COLLECTION_NAME = "NewsHW7"
CHROMA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chroma_news_hw7")
)
EMBEDDING_MODEL = "text-embedding-3-small"

# Semantic query used to surface "interesting" candidate articles
INTERESTING_QUERY = (
    "Breaking significant unusual high-impact business news major developments "
    "surprising earnings litigation regulatory changes leadership strategy"
)


def _get_openai_key() -> str:
    return st.secrets["openai_api_key"]


def load_news_collection():
    """Load existing Chroma collection (built offline). No embedding of CSV at app start."""
    if "hw7_news_collection" in st.session_state:
        return st.session_state.hw7_news_collection

    api_key = _get_openai_key()
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBEDDING_MODEL,
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=openai_ef)
    except Exception:
        st.session_state.hw7_news_collection = None
        return None

    st.session_state.hw7_news_collection = collection
    return collection


# ---------------------------------------------------------------------------
# Tool-style functions (used by the app; LLM receives their output in the prompt)
# ---------------------------------------------------------------------------

def search_news(
    query: str,
    collection,
    n_results: int = 12,
    company_filter: str | None = None,
) -> list[dict]:
    """
    Vector search over indexed news. Optional exact metadata filter on company_name.
    Returns list of {text, article_id, company_name, date, url, chunk_index}.
    """
    if collection is None or collection.count() == 0:
        return []

    n = min(max(1, n_results), collection.count())
    kwargs = {
        "query_texts": [query],
        "n_results": n,
        "include": ["documents", "metadatas", "distances"],
    }
    if company_filter and company_filter.strip():
        # Chroma requires exact match for $eq
        kwargs["where"] = {"company_name": {"$eq": company_filter.strip()}}

    results = collection.query(**kwargs)
    out: list[dict] = []
    if not results or not results.get("documents") or not results["documents"][0]:
        return out

    for i, doc in enumerate(results["documents"][0]):
        meta = (results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {}) or {}
        dist = None
        if results.get("distances") and results["distances"][0]:
            dist = results["distances"][0][i]
        out.append(
            {
                "text": doc,
                "article_id": meta.get("article_id", ""),
                "company_name": meta.get("company_name", ""),
                "date": meta.get("date", ""),
                "url": meta.get("url", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "distance": dist,
            }
        )
    return out


def retrieve_interesting_candidates(
    collection,
    n_chunks: int = 60,
    max_articles: int = 12,
) -> list[dict]:
    """
    Retrieve many chunks with a broad 'interesting' semantic query, dedupe by article_id,
    keep best-scoring chunk per article, return ranked list (best distance first).
    """
    if collection is None or collection.count() == 0:
        return []

    n = min(n_chunks, collection.count())
    results = collection.query(
        query_texts=[INTERESTING_QUERY],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    if not results or not results.get("documents") or not results["documents"][0]:
        return []

    best_by_article: dict[str, dict] = {}
    for i, doc in enumerate(results["documents"][0]):
        meta = (results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {}) or {}
        aid = meta.get("article_id", "")
        dist = results["distances"][0][i] if results.get("distances") and results["distances"][0] else None
        row = {
            "text": doc,
            "article_id": aid,
            "company_name": meta.get("company_name", ""),
            "date": meta.get("date", ""),
            "url": meta.get("url", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "distance": dist,
        }
        if aid not in best_by_article:
            best_by_article[aid] = row
        else:
            old_d = best_by_article[aid].get("distance")
            if dist is not None and (old_d is None or dist < old_d):
                best_by_article[aid] = row

    ranked = sorted(
        best_by_article.values(),
        key=lambda x: (x.get("distance") is None, x.get("distance") or 0.0),
    )
    return ranked[:max_articles]


def format_hits_for_prompt(hits: list[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(
            f"### Hit {i}\n"
            f"- company: {h.get('company_name', '')}\n"
            f"- date: {h.get('date', '')}\n"
            f"- url: {h.get('url', '')}\n"
            f"- article_id: {h.get('article_id', '')}\n"
            f"Excerpt:\n{h.get('text', '')}\n"
        )
    return "\n".join(lines) if lines else "(No matching articles in the index.)"


def detect_interesting_intent(message: str) -> bool:
    t = message.lower()
    patterns = (
        r"\bmost interesting\b",
        r"\binteresting news\b",
        r"\btop news\b",
        r"\bbest news\b",
        r"\bwhat'?s interesting\b",
        r"\bfind the most interesting\b",
        r"\brank(ed)?\b.*\b(news|articles|stories)\b",
    )
    return any(re.search(p, t) for p in patterns)


def extract_company_guess(message: str) -> str | None:
    """Lightweight 'news about X' — user may say 'about Toyota' or 'JPMorgan news'."""
    m = re.search(r"\babout\s+([^?.!]+)", message, re.I)
    if m:
        return m.group(1).strip().strip('"').strip("'")[:200]
    m = re.search(r"\bnews\s+(?:on|for|regarding)\s+([^?.!]+)", message, re.I)
    if m:
        return m.group(1).strip()[:200]
    return None


SYSTEM_BASE = """You are a news assistant. You MUST only use information from the article excerpts provided below.
If the excerpts do not contain enough information, say so clearly.
Always cite which article(s) you used (company name and/or URL when available).
Do not invent facts or cite sources outside the provided excerpts."""


def app():
    st.title("HW7 – News reporting bot (CSV corpus only)")
    st.caption(
        "Answers use only articles indexed from your CSV. "
        "Build the index once with: `python scripts/build_news_index.py`"
    )

    collection = load_news_collection()
    if collection is None:
        st.error(
            f"News index not found or empty. Set `OPENAI_API_KEY`, place `news.csv` in `data/`, "
            f"then run from project root:\n\n`python scripts/build_news_index.py`\n\n"
            f"Chroma path: `{CHROMA_PATH}`"
        )
        st.stop()

    n_chunks = collection.count()
    st.sidebar.metric("Indexed chunks", n_chunks)

    model_choice = st.sidebar.selectbox(
        "LLM (compare low vs high cost)",
        ("gpt-4o-mini (lower cost)", "gpt-4o (higher cost)"),
        key="hw7_model",
    )
    model_to_use = "gpt-4o-mini" if "mini" in model_choice else "gpt-4o"

    if "client" not in st.session_state:
        st.session_state.client = OpenAI(api_key=_get_openai_key())
    client = st.session_state.client

    if "hw7_messages" not in st.session_state:
        st.session_state.hw7_messages = [
            {
                "role": "assistant",
                "content": (
                    "Ask me about the news in your dataset. Examples:\n"
                    "- **Find the most interesting news** — I’ll rank articles with brief reasons.\n"
                    "- **Find news about Toyota** (or any topic/company) — I’ll pull relevant articles.\n"
                    "- Any follow-up question about those stories."
                ),
            }
        ]

    for msg in st.session_state.hw7_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about the news..."):
        st.session_state.hw7_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Decide retrieval strategy (tool-like functions)
        company_exact = None
        cg = extract_company_guess(prompt)

        if detect_interesting_intent(prompt):
            hits = retrieve_interesting_candidates(collection, n_chunks=80, max_articles=15)
            context = format_hits_for_prompt(hits)
            extra = (
                "The user asked for the MOST INTERESTING news in ranked order. "
                "Using ONLY the hits below, produce a numbered list from most to least interesting. "
                "For each item, give 1–2 sentences explaining WHY it is interesting (impact, novelty, conflict, scale). "
                "Include company name, date, and URL for each when available."
            )
        else:
            # Topic / company search: use query text; optional company filter if message looks like "only X"
            q = prompt
            if cg:
                q = f"{cg} {prompt}"
            hits = search_news(q, collection, n_results=15, company_filter=company_exact)
            if not hits and cg:
                hits = search_news(cg, collection, n_results=15)
            context = format_hits_for_prompt(hits)
            extra = (
                "Answer the user's question using ONLY the hits below. "
                "If multiple articles relate, summarize and cite each with URL. "
                "If nothing matches, say the corpus doesn't contain relevant articles."
            )

        system_content = f"{SYSTEM_BASE}\n\n{extra}\n\n## Retrieved article excerpts\n\n{context}"

        messages = [{"role": "system", "content": system_content}]
        messages.extend(st.session_state.hw7_messages[-10:])

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                stream=True,
                temperature=0.3,
            )
            response = st.write_stream(stream)

        st.session_state.hw7_messages.append({"role": "assistant", "content": response})

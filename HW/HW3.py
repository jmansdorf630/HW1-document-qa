import re
import streamlit as st
from openai import OpenAI
import requests

# Max tokens to pass to the LLM (token-based buffer limit).
max_tokens = 1000
# Conversation memory: keep last N messages (3 user–assistant exchanges).
MEMORY_BUFFER_SIZE = 6


def count_tokens(messages):
    """Approximate total tokens for chat messages."""
    total = 3  # reply priming
    for m in messages:
        total += 4 + len((m.get("content") or "") or "") + len(m.get("role", ""))
    return total // 4


# Browser-like headers so servers that block scripts/bots still allow the request.
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def get_url_text(url):
    """Fetch URL and return plain text (strip HTML tags). Never discarded—used in system prompt."""
    if not url or not str(url).strip():
        return ""
    try:
        r = requests.get(
            str(url).strip(),
            timeout=15,
            headers=REQUEST_HEADERS,
            allow_redirects=True,
        )
        r.raise_for_status()
        if r.encoding is None:
            r.encoding = "utf-8"
        text = r.text
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return (text[:15000] + "...") if len(text) > 15000 else text
    except Exception as e:
        return f"[Could not load URL: {e!s}]"


def is_yes(text):
    t = (text or "").strip().lower()
    return t in ("yes", "y", "yeah", "yep", "sure", "more", "please")


def is_no(text):
    t = (text or "").strip().lower()
    return t in ("no", "n", "nope", "nah", "no thanks")


def _openai_stream_to_text(stream):
    """Yield content strings from OpenAI chat completion stream for st.write_stream."""
    for chunk in stream:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                yield delta.content


def app():
    # Show title and description.
    st.title("MY question answering chatbot")
    st.write(
        "This chatbot answers questions in a simple, kid-friendly way so a 10-year-old can understand. "
        "You can provide one or two document URLs in the sidebar; their content is used as context and is never discarded. "
        "**Conversation memory:** a sliding buffer of the last 6 messages (3 user–assistant exchanges) is kept so the assistant can follow recent context. "
        'The assistant often asks "Do you want more info?" so you can request more detail on the same topic.'
    )

    # Resolve API key (required for this app; only OpenAI is used for now).
    api_key = None
    try:
        api_key = st.secrets.get("openai_api_key")
    except Exception:
        pass
    if not api_key:
        st.info(
            "Add your OpenAI API key in `.streamlit/secrets.toml` as `openai_api_key` to use this chatbot.",
            icon="🗝️",
        )
        return

    # Sidebar options
    st.sidebar.header("Options")
    st.sidebar.subheader("LLM Settings")
    llm_vendor = st.sidebar.selectbox("Select LLM Vendor", ("OpenAI", "Anthropic"))

    # Only OpenAI is implemented; Anthropic selection falls back to OpenAI to avoid 404.
    if llm_vendor == "OpenAI":
        model_to_use = "gpt-4o"
        st.sidebar.info("Using GPT-4o (latest premium)")
    else:
        model_to_use = "gpt-4o"
        st.sidebar.warning("Anthropic not configured here; using GPT-4o.")

    st.sidebar.write(f"**Model:** {model_to_use}")
    st.sidebar.subheader("Document URLs")
    url1 = st.sidebar.text_input("URL 1 (optional)", placeholder="https://example.com")
    url2 = st.sidebar.text_input("URL 2 (optional)", placeholder="https://example.com")

    # Build document context from sidebar URLs and show read status.
    _context_parts = []
    if url1:
        doc1 = get_url_text(url1)
        _context_parts.append("Document 1:\n" + doc1)
        if doc1.startswith("[Could not load"):
            st.sidebar.error("URL 1: could not load")
        else:
            st.sidebar.success(f"URL 1: read {len(doc1):,} chars")
    if url2:
        doc2 = get_url_text(url2)
        _context_parts.append("Document 2:\n" + doc2)
        if doc2.startswith("[Could not load"):
            st.sidebar.error("URL 2: could not load")
        else:
            st.sidebar.success(f"URL 2: read {len(doc2):,} chars")
    document_context = "\n\n".join(_context_parts) if _context_parts else "No documents provided."

    kid_friendly_system = (
        "You explain things in a simple, friendly way so that a 10-year-old can understand. "
        "Use short sentences and everyday words. When you answer a question, give a clear answer "
        "and then ask: 'Do you want more info?' When the user wants more info, give more details "
        "on the same topic in the same simple style, then ask again: 'Do you want more info?'"
    )
    system_with_context = (
        kid_friendly_system
        + "\n\nUse the following as context when answering. Do not discard this.\n\n"
        + document_context
    )

    if "phase" not in st.session_state:
        st.session_state.phase = "ask_question"
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""
    if "client" not in st.session_state:
        st.session_state.client = OpenAI(api_key=api_key)
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "What would you like to know? Ask me anything!"}
        ]

    for msg in st.session_state.messages:
        chat_msg = st.chat_message(msg["role"])
        chat_msg.write(msg["content"])

    if prompt := st.chat_input("Enter a message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        client = st.session_state.client
        phase = st.session_state.phase

        if phase in ("answered_ask_more", "gave_more_ask_again") and is_no(prompt):
            reply = "Sure! What else can I help you with?"
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.session_state.phase = "ask_question"
            st.session_state.last_question = ""
        elif phase in ("answered_ask_more", "gave_more_ask_again") and is_yes(prompt):
            messages_for_llm = [{"role": "system", "content": system_with_context}]
            messages_for_llm.extend(st.session_state.messages[-MEMORY_BUFFER_SIZE:])
            messages_for_llm.append({
                "role": "user",
                "content": "The user said they want more information. Give more details about what we were just talking about, in the same simple way. Then end by asking: Do you want more info?",
            })
            messages_to_send = messages_for_llm
            if count_tokens(messages_to_send) > max_tokens:
                messages_to_send = [messages_to_send[0]] + messages_to_send[-4:]
            st.caption(f"Tokens sent to LLM: {count_tokens(messages_to_send)} / {max_tokens}")
            stream = client.chat.completions.create(
                model=model_to_use,
                messages=messages_to_send,
                stream=True,
            )
            with st.chat_message("assistant"):
                response = st.write_stream(_openai_stream_to_text(stream))
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.phase = "gave_more_ask_again"
        else:
            if phase == "ask_question" or not (is_yes(prompt) or is_no(prompt)):
                st.session_state.last_question = prompt
            messages_for_llm = [{"role": "system", "content": system_with_context}]
            messages_for_llm.extend(st.session_state.messages[-MEMORY_BUFFER_SIZE:])
            st.caption(f"Tokens sent to LLM: {count_tokens(messages_for_llm)} / {max_tokens}")
            stream = client.chat.completions.create(
                model=model_to_use,
                messages=messages_for_llm,
                stream=True,
            )
            with st.chat_message("assistant"):
                response = st.write_stream(_openai_stream_to_text(stream))
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.phase = "answered_ask_more"

        st.session_state.messages = st.session_state.messages[-MEMORY_BUFFER_SIZE:]

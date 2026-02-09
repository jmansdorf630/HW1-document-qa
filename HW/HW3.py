import re
import streamlit as st
from openai import OpenAI
import requests

# Show title and description.
st.title("MY question answering chatbot")

st.write(
    "This chatbot answers questions in a simple, kid-friendly way so a 10-year-old can understand. "
    "You can provide one or two document URLs in the sidebar; their content is used as context and is never discarded. "
    "**Conversation memory:** a sliding buffer of the last 6 messages (3 user–assistant exchanges) is kept so the assistant can follow recent context. "
    "The assistant often asks \"Do you want more info?\" so you can request more detail on the same topic."
)

# Max tokens to pass to the LLM (token-based buffer limit).
max_tokens = 1000

# Conversation memory: keep last N messages (3 user–assistant exchanges).
MEMORY_BUFFER_SIZE = 6

# Sidebar options
st.sidebar.header("Options")

# LLM Vendor and Model Selection
st.sidebar.subheader("LLM Settings")
llm_vendor = st.sidebar.selectbox("Select LLM Vendor", 
                                   ("OpenAI", "Anthropic"))

if llm_vendor == "OpenAI":
    model_to_use = "gpt-4o"  # Latest premium OpenAI model
    st.sidebar.info("Using GPT-4o (latest premium)")
else:  # Anthropic
    model_to_use = "claude-opus-4-1"  # Latest premium Anthropic model
    st.sidebar.info("Using Claude Opus 4.1 (latest premium)")

# Model display in sidebar
st.sidebar.write(f"**Model:** {model_to_use}")

# URL input options
st.sidebar.subheader("Document URLs")
url1 = st.sidebar.text_input("URL 1 (optional)", placeholder="https://example.com")
url2 = st.sidebar.text_input("URL 2 (optional)", placeholder="https://example.com")

# Approximate token count (~4 chars per token for English; no tiktoken dependency).
def count_tokens(messages):
    """Approximate total tokens for chat messages."""
    total = 3  # reply priming
    for m in messages:
        total += 4 + len((m.get("content") or "") or "") + len(m.get("role", ""))
    return total // 4


def get_url_text(url):
    """Fetch URL and return plain text (strip HTML tags). Never discarded—used in system prompt."""
    if not url or not str(url).strip():
        return ""
    try:
        r = requests.get(str(url).strip(), timeout=10)
        r.raise_for_status()
        text = r.text
        # Strip HTML tags for readability
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return (text[:15000] + "...") if len(text) > 15000 else text
    except Exception:
        return "[Could not load URL.]"


# Build document context from sidebar URLs (used in system prompt, never discarded).
_context_parts = []
if url1:
    _context_parts.append("Document 1:\n" + get_url_text(url1))
if url2:
    _context_parts.append("Document 2:\n" + get_url_text(url2))
DOCUMENT_CONTEXT = "\n\n".join(_context_parts) if _context_parts else "No documents provided."

# System prompt: answer like for a 10-year-old and follow the "more info?" flow.
# URL context is included here and is never discarded.
KID_FRIENDLY_SYSTEM = (
    "You explain things in a simple, friendly way so that a 10-year-old can understand. "
    "Use short sentences and everyday words. When you answer a question, give a clear answer "
    "and then ask: 'Do you want more info?' When the user wants more info, give more details "
    "on the same topic in the same simple style, then ask again: 'Do you want more info?'"
)
SYSTEM_WITH_CONTEXT = (
    KID_FRIENDLY_SYSTEM
    + "\n\nUse the following as context when answering. Do not discard this.\n\n"
    + DOCUMENT_CONTEXT
)

# Conversation phase: "ask_question" | "answered_ask_more" | "gave_more_ask_again"
if "phase" not in st.session_state:
    st.session_state.phase = "ask_question"
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

def is_yes(text):
    t = (text or "").strip().lower()
    return t in ("yes", "y", "yeah", "yep", "sure", "more", "please")

def is_no(text):
    t = (text or "").strip().lower()
    return t in ("no", "n", "nope", "nah", "no thanks")

# Create an openAI client
if "client" not in st.session_state:
    api_key = st.secrets["openai_api_key"]
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
    last_question = st.session_state.last_question

    # ---- User said "No" (after we asked "Do you want more info?") → back to help ----
    if phase in ("answered_ask_more", "gave_more_ask_again") and is_no(prompt):
        reply = "Sure! What else can I help you with?"
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state.phase = "ask_question"
        st.session_state.last_question = ""
    # ---- User said "Yes" (want more info) → provide more, then ask again ----
    elif phase in ("answered_ask_more", "gave_more_ask_again") and is_yes(prompt):
        # System prompt with URL context (never discarded) + 6-message buffer.
        messages_for_llm = [{"role": "system", "content": SYSTEM_WITH_CONTEXT}]
        messages_for_llm.extend(st.session_state.messages[-MEMORY_BUFFER_SIZE:])
        messages_for_llm.append({
            "role": "user",
            "content": "The user said they want more information. Give more details about what we were just talking about, in the same simple way. Then end by asking: Do you want more info?",
        })
        messages_to_send = messages_for_llm
        if count_tokens(messages_to_send) > max_tokens:
            messages_to_send = [messages_to_send[0]] + messages_to_send[-4:]
        tokens_this_request = count_tokens(messages_to_send)
        st.caption(f"Tokens sent to LLM: {tokens_this_request} / {max_tokens}")
        stream = client.chat.completions.create(
            model=model_to_use,
            messages=messages_to_send,
            stream=True,
        )
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.phase = "gave_more_ask_again"
    # ---- New question (or unclear reply): answer then ask "Do you want more info?" ----
    else:
        if phase == "ask_question" or not (is_yes(prompt) or is_no(prompt)):
            st.session_state.last_question = prompt
        # System prompt with URL context (never discarded) + 6-message buffer.
        messages_for_llm = [{"role": "system", "content": SYSTEM_WITH_CONTEXT}]
        messages_for_llm.extend(st.session_state.messages[-MEMORY_BUFFER_SIZE:])
        tokens_this_request = count_tokens(messages_for_llm)
        st.caption(f"Tokens sent to LLM: {tokens_this_request} / {max_tokens}")
        stream = client.chat.completions.create(
            model=model_to_use,
            messages=messages_for_llm,
            stream=True,
        )
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.phase = "answered_ask_more"

    # 6-message buffer: keep only last MEMORY_BUFFER_SIZE messages for next turn
    st.session_state.messages = st.session_state.messages[-MEMORY_BUFFER_SIZE:]
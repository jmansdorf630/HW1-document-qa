"""
HW5 – Short-term memory chatbot that answers questions about documents.
Uses relevant_course_info and relevant_club_info to retrieve from ChromaDB,
then invokes the LLM with those results (system prompt).
"""
import streamlit as st
from openai import OpenAI
import tiktoken

from HW.HW4 import create_lab4_vectordb, create_su_orgs_vectordb


# ---------------------------------------------------------------------------
# Retrieval functions: take a query and return relevant info from ChromaDB
# (embedding is computed from the query inside the collection's embedding_function)
# ---------------------------------------------------------------------------

def relevant_course_info(query: str, collection, n_results: int = 5) -> str:
    """
    Takes a query (from the user or LLM) and returns relevant information
    from the syllabus ChromaDB collection.
    """
    if collection is None or collection.count() == 0:
        return "(No documents in collection.)"
    n = min(n_results, collection.count())
    results = collection.query(
        query_texts=[query],
        n_results=n,
        include=["documents", "metadatas"],
    )
    parts = []
    if results and results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            meta = (results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {}) or {}
            src = meta.get("filename", "syllabus")
            parts.append(f"[Syllabus: {src}]\n{doc}")
    return "\n\n---\n\n".join(parts) if parts else "(No relevant passages found.)"


def relevant_club_info(query: str, collection, n_results: int = 6) -> str:
    """
    Takes a query and returns relevant information from the SU organizations
    ChromaDB collection.
    """
    if collection is None or collection.count() == 0:
        return "(No documents in collection.)"
    n = min(n_results, collection.count())
    results = collection.query(
        query_texts=[query],
        n_results=n,
        include=["documents", "metadatas"],
    )
    parts = []
    if results and results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            meta = (results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {}) or {}
            src = meta.get("source", "org")
            parts.append(f"[SU Org: {src}]\n{doc}")
    return "\n\n---\n\n".join(parts) if parts else "(No relevant passages found.)"


# --- HW5 page: short-term memory chatbot ---
MAX_INTERACTIONS = 5


def app():
    st.title("HW5 – Document QA (short-term memory)")

    vectordb = create_lab4_vectordb()
    if vectordb is None:
        st.warning(
            "Vector DB could not be loaded. Check that Lab-04-Data.zip is available "
            "and that the OpenAI API key is set in secrets."
        )
        st.stop()

    try:
        su_orgs_vectordb = create_su_orgs_vectordb()
        su_orgs_count = su_orgs_vectordb.count()
    except Exception:
        su_orgs_vectordb = None
        su_orgs_count = 0

    count = vectordb.count()
    caption = f"Vector DB ready with {count} syllabus documents"
    if su_orgs_vectordb and su_orgs_count:
        caption += f" and {su_orgs_count} SU orgs chunks. "
    caption += "Ask questions below; answers use retrieved excerpts."
    st.caption(caption)
    st.write("")

    max_tokens = 1000
    openai_model = st.sidebar.selectbox("Which Model?", ("mini", "regular"), key="hw5_model")
    model_to_use = "gpt-4o-mini" if openai_model == "mini" else "gpt-4o"

    try:
        encoding = tiktoken.encoding_for_model("gpt-4o")
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(messages):
        total = 3
        for m in messages:
            total += 4
            total += len(encoding.encode(m["role"]))
            total += len(encoding.encode(m.get("content", "") or ""))
        return total

    system_base = (
        "You explain in a simple, friendly way. When your answer is based on the provided excerpts, "
        "say so (e.g. 'Based on the course syllabi:' or 'According to the SU organization pages:'). "
        "When the answer is NOT in the excerpts, say so (e.g. 'The materials don't mention this; here's what I know:'). "
        "Prefer SU organization excerpts for questions about organizations, clubs, or student groups."
    )

    if "hw5_phase" not in st.session_state:
        st.session_state.hw5_phase = "ask_question"
    if "hw5_last_question" not in st.session_state:
        st.session_state.hw5_last_question = ""

    def _trim_to_last_n_interactions(messages, n=MAX_INTERACTIONS):
        max_messages = n * 2 + 1
        if len(messages) <= max_messages:
            return messages
        prefix = [messages[0]] if messages and messages[0]["role"] == "assistant" else []
        return prefix + messages[-(n * 2):]

    def is_yes(text):
        t = (text or "").strip().lower()
        return t in ("yes", "y", "yeah", "yep", "sure", "more", "please")

    def is_no(text):
        t = (text or "").strip().lower()
        return t in ("no", "n", "nope", "nah", "no thanks")

    if "hw5_messages" not in st.session_state:
        st.session_state.hw5_messages = [
            {"role": "assistant", "content": "What would you like to know about courses or organizations? Ask me anything!"}
        ]

    if "client" not in st.session_state:
        st.session_state.client = OpenAI(api_key=st.secrets["openai_api_key"])
    client = st.session_state.client

    for msg in st.session_state.hw5_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Enter a message"):
        st.session_state.hw5_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        phase = st.session_state.hw5_phase

        # User said "No" → back to help
        if phase in ("answered_ask_more", "gave_more_ask_again") and is_no(prompt):
            reply = "Sure! What else can I help you with?"
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.hw5_messages.append({"role": "assistant", "content": reply})
            st.session_state.hw5_phase = "ask_question"
            st.session_state.hw5_last_question = ""
        # User said "Yes" (want more info)
        elif phase in ("answered_ask_more", "gave_more_ask_again") and is_yes(prompt):
            messages_for_llm = [{"role": "system", "content": system_base}]
            messages_for_llm.extend(st.session_state.hw5_messages[-6:])
            messages_for_llm.append({
                "role": "user",
                "content": "The user said they want more information. Give more details about what we were just talking about, in the same simple way. Then end by asking: Do you want more info?",
            })
            if count_tokens(messages_for_llm) > max_tokens:
                messages_for_llm = [messages_for_llm[0]] + messages_for_llm[-4:]
            st.caption(f"Tokens sent to LLM: {count_tokens(messages_for_llm)} / {max_tokens}")
            stream = client.chat.completions.create(
                model=model_to_use,
                messages=messages_for_llm,
                stream=True,
            )
            with st.chat_message("assistant"):
                response = st.write_stream(stream)
            st.session_state.hw5_messages.append({"role": "assistant", "content": response})
            st.session_state.hw5_phase = "gave_more_ask_again"
        # New question: use retrieval functions, then invoke LLM with results
        else:
            if phase == "ask_question" or not (is_yes(prompt) or is_no(prompt)):
                st.session_state.hw5_last_question = prompt

            # Retrieve via the required functions (query → ChromaDB → context string)
            course_context = relevant_course_info(prompt, vectordb, n_results=3)
            club_context = relevant_club_info(prompt, su_orgs_vectordb, n_results=6) if su_orgs_vectordb else ""
            context_parts = [course_context]
            if club_context and "No documents" not in club_context and "No relevant" not in club_context:
                context_parts.append(club_context)
            context_text = "\n\n---\n\n".join(context_parts)

            system_with_context = (
                system_base
                + "\n\nExcerpts (use these when they answer the question):\n"
                + context_text
            )

            messages_for_llm = [{"role": "system", "content": system_with_context}]
            messages_for_llm.extend(st.session_state.hw5_messages)
            while count_tokens(messages_for_llm) > max_tokens and len(messages_for_llm) > 3:
                if messages_for_llm[1]["role"] == "assistant":
                    messages_for_llm = [messages_for_llm[0]] + messages_for_llm[3:]
                else:
                    messages_for_llm = [messages_for_llm[0]] + messages_for_llm[2:]
            st.caption(f"Tokens sent to LLM: {count_tokens(messages_for_llm)} / {max_tokens}")
            stream = client.chat.completions.create(
                model=model_to_use,
                messages=messages_for_llm,
                stream=True,
            )
            with st.chat_message("assistant"):
                response = st.write_stream(stream)
            st.session_state.hw5_messages.append({"role": "assistant", "content": response})
            st.session_state.hw5_phase = "answered_ask_more"

        st.session_state.hw5_messages = _trim_to_last_n_interactions(st.session_state.hw5_messages)

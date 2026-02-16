import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import zipfile
from pypdf import PdfReader
from io import BytesIO
import os
import glob
import tiktoken
from bs4 import BeautifulSoup


def create_lab4_vectordb():
    """
    Construct a ChromaDB collection named "Lab4Collection" with PDF documents.
    Uses OpenAI embeddings model (text-embedding-3-small).
    Stores the collection in st.session_state.Lab4_VectorDB to avoid recreating it.
    """
    # Check if vector database already exists in session state
    if "Lab4_VectorDB" in st.session_state:
        return st.session_state.Lab4_VectorDB
    
    # Initialize OpenAI client
    if "client" not in st.session_state:
        api_key = st.secrets["openai_api_key"]
        st.session_state.client = OpenAI(api_key=api_key)
    
    client = st.session_state.client
    
    # Create OpenAI embedding function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=st.secrets["openai_api_key"],
        model_name="text-embedding-3-small"
    )
    
    # Initialize ChromaDB client (persistent storage)
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Create or get the collection
    try:
        collection = chroma_client.get_collection(
            name="Lab4Collection",
            embedding_function=openai_ef
        )
        # Collection exists, check if it has documents
        if collection.count() > 0:
            st.session_state.Lab4_VectorDB = collection
            return collection
    except Exception:
        # Collection doesn't exist, create it
        collection = chroma_client.create_collection(
            name="Lab4Collection",
            embedding_function=openai_ef
        )
    
    # Path to the zip file: try project root, data/, script-relative, then ~/Downloads
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    possible_paths = [
        "Lab-04-Data.zip",
        "data/Lab-04-Data.zip",
        os.path.join(project_root, "Lab-04-Data.zip"),
        os.path.join(project_root, "data", "Lab-04-Data.zip"),
        os.path.expanduser("~/Downloads/Lab-04-Data.zip"),
    ]
    zip_path = None
    for p in possible_paths:
        if os.path.isfile(p):
            zip_path = p
            break
    if zip_path is None:
        st.error(
            "**Lab-04-Data.zip** not found. "
            "For **Streamlit Cloud**: add the zip to the repo (project root or `data/` folder). "
            "For **local**: put it in your project root, in a `data/` folder, or in **Downloads**."
        )
        return None
    
    # List of PDF files to process
    pdf_files = [
        "IST 488 Syllabus - Building Human-Centered AI Applications.pdf",
        "IST 314 Syllabus - Interacting with AI.pdf",
        "IST 343 Syllabus - Data in Society.pdf",
        "IST 256 Syllabus - Intro to Python for the Information Profession.pdf",
        "IST 387 Syllabus - Introduction to Applied Data Science.pdf",
        "IST 418 Syllabus - Big Data Analytics.pdf",
        "IST 195 Syllabus - Information Technologies.pdf"
    ]
    
    # Extract and process PDFs from zip file
    documents = []
    metadatas = []
    ids = []
    
    try:
        zip_ref = zipfile.ZipFile(zip_path, 'r')
    except FileNotFoundError:
        st.error(f"Zip file not found: {zip_path}")
        return None
    except OSError as e:
        st.error(f"Cannot open zip file: {e}")
        return None

    with zip_ref:
        for pdf_filename in pdf_files:
            # Construct the full path within the zip
            zip_internal_path = f"Lab-04-Data/{pdf_filename}"
            
            try:
                # Read PDF from zip
                pdf_bytes = zip_ref.read(zip_internal_path)
                pdf_reader = PdfReader(BytesIO(pdf_bytes))
                
                # Extract text from all pages
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
                
                # Clean up text (remove excessive whitespace)
                text_content = " ".join(text_content.split())
                
                if text_content.strip():  # Only add if text was extracted
                    documents.append(text_content)
                    metadatas.append({
                        "filename": pdf_filename,
                        "source": "Lab-04-Data"
                    })
                    ids.append(pdf_filename)  # Use filename as unique ID
                    
            except KeyError:
                st.warning(f"PDF file not found in zip: {pdf_filename}")
                continue
            except Exception as e:
                st.error(f"Error processing {pdf_filename}: {str(e)}")
                continue
    
    # Add documents to ChromaDB collection
    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        st.session_state.Lab4_VectorDB = collection
        st.success(f"Successfully created ChromaDB collection with {len(documents)} documents!")
    else:
        st.error("No documents were successfully processed from the PDF files.")
        return None
    
    return collection


# ---------------------------------------------------------------------------
# SU Orgs vector DB: HTML files as RAG sources
# (a)(i) Copy all HTML pages from the zipped folder to the project area.
# (a)(ii) Build vector DB with those documents; chunk each into two mini-documents.
# (b) Create the vector DB only if it does not already exist.
# ---------------------------------------------------------------------------

def _copy_su_orgs_html_to_project():
    """Copy all HTML files from su_orgs zip into project's su_orgs/ folder. No-op if su_orgs/ already has HTML."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    su_orgs_dir = os.path.join(project_root, "su_orgs")
    if os.path.isdir(su_orgs_dir) and glob.glob(os.path.join(su_orgs_dir, "*.html")):
        return su_orgs_dir
    zip_paths = [
        os.path.join(project_root, "su_orgs (1).zip"),
        os.path.join(project_root, "data", "su_orgs (1).zip"),
        os.path.expanduser(os.path.join("~/Downloads", "su_orgs (1).zip")),
    ]
    zip_path = None
    for p in zip_paths:
        if os.path.isfile(p):
            zip_path = p
            break
    if not zip_path:
        return su_orgs_dir
    os.makedirs(su_orgs_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith(".html"):
                zf.extract(name, project_root)
    return su_orgs_dir


def _extract_text_from_html(file_path):
    """Extract main text from an HTML file; strip scripts and styles."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except Exception:
        return ""
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    body = soup.find("body") or soup
    text = body.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def _chunk_document_into_two(text):
    """
    Chunking method: Fixed-size midpoint (binary) chunking.
    We split the document text at the character midpoint into exactly two chunks:
    first half = chunk 0, second half = chunk 1.

    Why this method:
    - The assignment requires exactly two mini-documents per source; midpoint splitting
      guarantees exactly two chunks regardless of document length.
    - It is simple and deterministic (no dependency on sentence boundaries or tokenizers).
    - For typical organization HTML pages, the first half often has title/nav/overview
      and the second half body/details, so retrieval can still find the relevant half.
    - Other methods (e.g. sentence-based or overlap) would need extra logic to force
      exactly two chunks; midpoint is direct.
    """
    text = (text or "").strip()
    if not text:
        return ["", ""]
    n = len(text)
    mid = (n + 1) // 2
    return [text[:mid].strip(), text[mid:].strip()]


SU_ORGS_COLLECTION_NAME = "SuOrgsCollection"


def create_su_orgs_vectordb():
    """
    Build a ChromaDB collection from all HTML files in su_orgs/ (RAG for SU organizations).
    Creates the vector DB file/collection only if it does not already exist, so the app
    can be run multiple times without rebuilding.
    """
    if "SuOrgs_VectorDB" in st.session_state:
        return st.session_state.SuOrgs_VectorDB

    if "client" not in st.session_state:
        st.session_state.client = OpenAI(api_key=st.secrets["openai_api_key"])
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=st.secrets["openai_api_key"],
        model_name="text-embedding-3-small",
    )
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # (b) Create the vector DB only if it does not already exist
    try:
        collection = chroma_client.get_collection(
            name=SU_ORGS_COLLECTION_NAME,
            embedding_function=openai_ef,
        )
        if collection.count() > 0:
            st.session_state.SuOrgs_VectorDB = collection
            return collection
        chroma_client.delete_collection(SU_ORGS_COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=SU_ORGS_COLLECTION_NAME,
        embedding_function=openai_ef,
    )

    # (a)(i) Copy HTML from zip to project if needed
    _copy_su_orgs_html_to_project()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    su_orgs_dir = os.path.join(project_root, "su_orgs")
    if not os.path.isdir(su_orgs_dir):
        st.session_state.SuOrgs_VectorDB = collection
        return collection

    # (a)(ii) Chunk each document into two mini-documents and add to collection
    html_files = sorted(glob.glob(os.path.join(su_orgs_dir, "*.html")))
    documents = []
    metadatas = []
    ids = []
    for fp in html_files:
        text = _extract_text_from_html(fp)
        if not text.strip():
            continue
        chunks = _chunk_document_into_two(text)
        name = os.path.basename(fp)
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            documents.append(chunk)
            metadatas.append({"source": name, "chunk_index": i})
            ids.append(f"{name}__chunk_{i}")

    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
    st.session_state.SuOrgs_VectorDB = collection
    return collection


# --- HW4 page UI: Course information chatbot (RAG) ---
MAX_INTERACTIONS = 5


def app():
    st.title("HW4 – Course information chatbot")

    # Initialize both vector DBs when the page loads (each created only if not already existing)
    vectordb = create_lab4_vectordb()
    if vectordb is None:
        st.warning(
            "Vector DB could not be loaded. Check that the Lab-04-Data.zip file is available "
            "at the expected path and that the OpenAI API key is set in secrets."
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
        caption += f" and {su_orgs_count} SU orgs chunks (HTML). "
    caption += "Ask questions below—answers will cite when they use these materials."
    st.caption(caption)
    st.write("")  # spacing

    # --- Lab 3–style setup: token counting, model choice, phase, messages ---
    max_tokens = 1000
    openAI_model = st.sidebar.selectbox("Which Model?", ("mini", "regular"), key="lab4_model")
    model_to_use = "gpt-4o-mini" if openAI_model == "mini" else "gpt-4o"

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

    KID_FRIENDLY_SYSTEM = (
        "You explain things in a simple, friendly way so that a 10-year-old can understand. "
        "Use short sentences and everyday words. When you answer a question, give a clear answer "
        "and then ask: 'Do you want more info?' When the user wants more info, give more details "
        "on the same topic in the same simple style, then ask again: 'Do you want more info?'"
    )

    if "lab4_phase" not in st.session_state:
        st.session_state.lab4_phase = "ask_question"
    if "lab4_last_question" not in st.session_state:
        st.session_state.lab4_last_question = ""

    def _trim_to_last_n_interactions(messages, n=MAX_INTERACTIONS):
        """Keep at most the last n interactions (each = user + assistant pair)."""
        max_messages = n * 2 + 1  # 1 optional initial greeting + n pairs
        if len(messages) <= max_messages:
            return messages
        prefix = [messages[0]] if messages and messages[0]["role"] == "assistant" else []
        return prefix + messages[-(n * 2) :]

    def is_yes(text):
        t = (text or "").strip().lower()
        return t in ("yes", "y", "yeah", "yep", "sure", "more", "please")

    def is_no(text):
        t = (text or "").strip().lower()
        return t in ("no", "n", "nope", "nah", "no thanks")

    if "lab4_messages" not in st.session_state:
        st.session_state.lab4_messages = [
            {"role": "assistant", "content": "What would you like to know about the organizations? Ask me anything!"}
        ]

    # Ensure OpenAI client exists for chat
    if "client" not in st.session_state:
        api_key = st.secrets["openai_api_key"]
        st.session_state.client = OpenAI(api_key=api_key)
    client = st.session_state.client

    # Render chat history
    for msg in st.session_state.lab4_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input and RAG + LLM flow
    if prompt := st.chat_input("Enter a message"):
        st.session_state.lab4_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        phase = st.session_state.lab4_phase
        last_question = st.session_state.lab4_last_question

        # ---- User said "No" → back to help ----
        if phase in ("answered_ask_more", "gave_more_ask_again") and is_no(prompt):
            reply = "Sure! What else can I help you with?"
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.lab4_messages.append({"role": "assistant", "content": reply})
            st.session_state.lab4_phase = "ask_question"
            st.session_state.lab4_last_question = ""
        # ---- User said "Yes" (want more info) ----
        elif phase in ("answered_ask_more", "gave_more_ask_again") and is_yes(prompt):
            messages_for_llm = [{"role": "system", "content": KID_FRIENDLY_SYSTEM}]
            messages_for_llm.extend(st.session_state.lab4_messages[-6:])
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
            st.session_state.lab4_messages.append({"role": "assistant", "content": response})
            st.session_state.lab4_phase = "gave_more_ask_again"
        # ---- New question: RAG (retrieve from vector DB) then answer ----
        else:
            if phase == "ask_question" or not (is_yes(prompt) or is_no(prompt)):
                st.session_state.lab4_last_question = prompt

            # Retrieve relevant chunks from syllabi and (if available) SU orgs HTML collections
            n_results_syllabi = 3
            n_results_su_orgs = 6  # Retrieve more org chunks so org questions get ample context
            context_parts = []
            results = vectordb.query(
                query_texts=[prompt],
                n_results=min(n_results_syllabi, vectordb.count()),
                include=["documents", "metadatas"],
            )
            if results and results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    meta = (results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {}) or {}
                    src = meta.get("filename", "syllabus")
                    context_parts.append(f"[Syllabus: {src}]\n{doc}")
            if su_orgs_vectordb and su_orgs_vectordb.count() > 0:
                su_results = su_orgs_vectordb.query(
                    query_texts=[prompt],
                    n_results=min(n_results_su_orgs, su_orgs_vectordb.count()),
                    include=["documents", "metadatas"],
                )
                if su_results and su_results["documents"] and su_results["documents"][0]:
                    for i, doc in enumerate(su_results["documents"][0]):
                        meta = (su_results["metadatas"][0][i] if su_results["metadatas"] and su_results["metadatas"][0] else {}) or {}
                        src = meta.get("source", "org")
                        context_parts.append(f"[SU Org: {src}]\n{doc}")
            context_text = "\n\n---\n\n".join(context_parts) if context_parts else "(No relevant passages found.)"

            # Prompt engineering: require the bot to be clear when using RAG vs general knowledge
            system_with_context = (
                KID_FRIENDLY_SYSTEM
                + "\n\nYou have access to the following excerpts from course syllabi and SU organization pages (retrieved by RAG). "
                "When your answer is based on these excerpts, you MUST say so clearly at the start, e.g. "
                "'Based on the course syllabi:' or 'According to the SU organization pages:' or 'According to the materials I have:'. "
                "When the answer is NOT in the excerpts (neither syllabi nor organization pages), you MUST say so clearly, e.g. "
                "'This isn’t in the syllabi or organization materials I have; from general knowledge:' or 'The materials don’t mention this; here’s what I know:'. "
                "PREFER using the SU organization excerpts when the user asks about organizations, clubs, or student groups. "
                "Keep answers simple and kid-friendly.\n\nExcerpts (use these when they answer the question):\n"
                + context_text
            )

            messages_for_llm = [{"role": "system", "content": system_with_context}]
            messages_for_llm.extend(st.session_state.lab4_messages)
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
            st.session_state.lab4_messages.append({"role": "assistant", "content": response})
            st.session_state.lab4_phase = "answered_ask_more"

        # (a) Trim conversation buffer to at most the last 5 interactions
        st.session_state.lab4_messages = _trim_to_last_n_interactions(st.session_state.lab4_messages)
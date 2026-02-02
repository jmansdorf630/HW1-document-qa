import streamlit as st
from openai import OpenAI, AuthenticationError
import requests
from bs4 import BeautifulSoup


def read_url_content(url: str):
    """Fetch and return the visible text content from a URL using requests + BeautifulSoup.

    Returns the extracted text or None on failure.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text(separator=" ", strip=True)
    except requests.RequestException as e:
        # Log to Streamlit when used in the app
        st.error(f"Error reading {url}: {e}")
        return None


# Show title and description.
st.title("🕸️ Document Summarizer (HW2)")
st.write(
    "Enter a URL below and generate a summary of the page content based on your selected options."
)

# Get the OpenAI API key (try secrets first, otherwise ask the user)
openai_api_key = st.secrets.get("openai_api_key") if hasattr(st, "secrets") else None
openai_api_key = st.text_input("OpenAI API Key", type="password", value=openai_api_key or "")

# New: get Anthropic key as well (only show input if Anthropic is selected)
anthropic_api_key = st.secrets.get("anthropic_api_key") if hasattr(st, "secrets") else None
if llm_provider == "Anthropic (Claude)":
    anthropic_api_key = st.text_input("Anthropic API Key", type="password", value=anthropic_api_key or "")

# Sidebar for summary options and language & model selection.
with st.sidebar:
    st.header("Summary Options")
    summary_type = st.radio(
        "Choose a summary type:",
        ("Summarize the document in 100 words",
         "Summarize the document in 2 connecting paragraphs",
         "Summarize the document in 5 bullet points")
    )

    # New: let the user pick the LLM provider
    llm_provider = st.selectbox(
        "LLM Provider",
        ("OpenAI", "Anthropic (Claude)")
    )

    language = st.selectbox("Output language", ("English", "Spanish", "French", "German", "Chinese (Simplified)", "Any"))
    use_advanced_model = st.checkbox("Use advanced model")

# Determine the model based on the checkbox and provider.
if llm_provider == "OpenAI":
    model = "gpt-4" if use_advanced_model else "gpt-3.5-turbo"
else:
    # Map "advanced" to a higher Claude model when Anthropic is selected
    model = "claude-3" if use_advanced_model else "claude-2"

# URL input (top of the screen, not in sidebar)
url = st.text_input("Enter a URL to summarize", placeholder="https://example.com")

# Summarize button triggers the workflow
if st.button("Summarize"):
    if not url:
        st.warning("Please enter a URL to summarize.")
    elif not openai_api_key:
        st.warning("Please add your OpenAI API key to continue.")
    else:
        with st.spinner("Fetching page content..."):
            document = read_url_content(url)
        if not document:
            st.error(f"Failed to read content from the URL: {url}")
        else:
            # Instruct the assistant about the requested output language.
            if language == "Any":
                system_content = (
                    "You are a helpful assistant. Respond in the language of the source document "
                    "when possible; otherwise respond in English."
                )
            else:
                system_content = f"You are a helpful assistant. Please produce the summary in {language}."

            # Small UI note when language selection is 'Any'
            if language == "Any":
                st.info("Language set to 'Any': the assistant will attempt to use the source document's language.")

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"{summary_type}: {document}"},
            ]

            # Create an OpenAI client and stream the summary to the page (or use Anthropic)
            try:
                if llm_provider == "OpenAI":
                    if not openai_api_key:
                        st.warning("Please add your OpenAI API key to continue.")
                    else:
                        client = OpenAI(api_key=openai_api_key)
                        stream = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            stream=True,
                        )
                        st.header("Document Summary")
                        st.write_stream(stream)

                else:  # Anthropic (Claude)
                    if not anthropic_api_key:
                        st.warning("Please add your Anthropic API key to continue.")
                    else:
                        # Build a simple prompt that includes the system instruction + user content.
                        prompt = f"{system_content}\n\nHuman: {summary_type}: {document}\n\nAssistant:"
                        headers = {
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {anthropic_api_key}",
                        }
                        payload = {
                            "model": model,
                            "prompt": prompt,
                            "max_tokens_to_sample": 1000,
                        }
                        resp = requests.post("https://api.anthropic.com/v1/complete", json=payload, headers=headers, timeout=30)
                        if resp.status_code != 200:
                            st.error(f"Anthropic API error ({resp.status_code}): {resp.text}")
                        else:
                            data = resp.json()
                            # Try common keys to find the completion text
                            completion = data.get("completion") or data.get("completion", "")
                            # Fallback to whole text if structure differs
                            completion_text = completion if isinstance(completion, str) else data.get("text") or str(data)
                            st.header("Document Summary")
                            st.write(completion_text)

            except AuthenticationError:
                st.error("Invalid OpenAI API key. Please check your API key and try again.")
            except Exception as e:
                st.error(f"An error occurred while summarizing: {e}")
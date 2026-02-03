import streamlit as st
from openai import OpenAI, AuthenticationError
import requests
from bs4 import BeautifulSoup
import os


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


def app():
    # Show title and description.
    st.title("🕸️ Document Summarizer (HW2)")
    st.write(
        "Enter a URL below and generate a summary of the page content based on your selected options."
    )

    # Do NOT pre-fill or store API keys in the app. Resolve them on-demand from environment/Streamlit secrets
    # or the user's ephemeral input (entered without pre-filling). This avoids leaking keys in the UI/session.
    # Keys are resolved later when the Summarize button is pressed.

    def _resolve_api_key(provider: str):
        """Return (api_key, session_key_name) using env -> Streamlit secrets -> session input."""
        key_name = "openai_api_key" if provider == "OpenAI" else "anthropic_api_key"
        session_key = "openai_input" if provider == "OpenAI" else "anthropic_input"
        env_key = os.environ.get(key_name.upper())
        secrets_key = getattr(st, "secrets", {}).get(key_name) if getattr(st, "secrets", None) else None
        session_value = st.session_state.get(session_key)
        return env_key or secrets_key or session_value, session_key

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

        # Show the relevant API key input based on selected provider (no pre-fill)
        if llm_provider == "OpenAI":
            st.info("OpenAI API key is read from the environment or Streamlit secrets. Enter a key below to use for this session (it will not be stored).")
            st.text_input("OpenAI API Key", type="password", key="openai_input")
        elif llm_provider == "Anthropic (Claude)":
            st.info("Anthropic API key is read from the environment or Streamlit secrets. Enter a key below to use for this session (it will not be stored).")
            st.text_input("Anthropic API Key", type="password", key="anthropic_input")

    # Determine the model based on the checkbox and provider.
    if llm_provider == "OpenAI":
        model = "gpt-4" if use_advanced_model else "gpt-3.5-turbo"
    else:
        model = "claude-3" if use_advanced_model else "claude-2"

    # URL input (top of the screen, not in sidebar)
    url = st.text_input("Enter a URL to summarize", placeholder="https://example.com")

    # Summarize button triggers the workflow
    if st.button("Summarize"):
        if not url:
            st.warning("Please enter a URL to summarize.")
        else:
            # Resolve API key on-demand (env/secrets override by ephemeral session input)
            api_key, session_key = _resolve_api_key(llm_provider)
            if not api_key:
                provider_name = "OpenAI" if llm_provider == "OpenAI" else "Anthropic (Claude)"
                st.warning(f"Please add your {provider_name} API key to continue.")
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
                        client = OpenAI(api_key=api_key)
                        stream = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            stream=True,
                        )
                        st.header("Document Summary")
                        st.write_stream(stream)

                    else:  # Anthropic (Claude)
                        # Build a simple prompt that includes the system instruction + user content.
                        prompt = f"{system_content}\n\nHuman: {summary_type}: {document}\n\nAssistant:"
                        headers = {
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key}",
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
                finally:
                    # Clear ephemeral session inputs so keys aren't left in session state
                    try:
                        if session_key in st.session_state:
                            st.session_state.pop(session_key, None)
                    except Exception:
                        # Don't raise if session state access fails; we only try to be defensive here.
                        pass


# expose runnable entrypoint for whatever loader expects it
if __name__ == "__main__":
    app()
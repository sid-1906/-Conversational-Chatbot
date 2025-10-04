import streamlit as st
import os
from langchain_huggingface import HuggingFaceHub
from typing import List, Union

# --- Configuration for Free Deployment (Hugging Face) ---

def get_hf_token() -> Union[str, None]:
    """Retrieves the HF_TOKEN securely from Streamlit secrets or environment."""
    # This token is required to authenticate with the free Hugging Face Inference API.
    token = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN"))
    if not token:
        # Provide a specific, friendly warning and instruction if the token is missing
        st.error("HUGGING FACE ACCESS TOKEN (HF_TOKEN) not found.")
        st.markdown(
            """
            **To run this app:**
            1. Sign up for a free Hugging Face account.
            2. Generate a **free User Access Token** (Settings -> Access Tokens).
            3. Set the secret in your Streamlit Cloud app settings as `HF_TOKEN`.
            """
        )
        st.stop()
    return token

@st.cache_resource
def get_hf_llm(token: str):
    """Initializes and caches the HuggingFaceHub model wrapper."""
    st.info("Initializing Hugging Face LLM... This only happens once.")
    
    # Using a popular, free, general-purpose model suitable for chat.
    return HuggingFaceHub(
        repo_id="google/flan-t5-large",
        huggingfacehub_api_token=token,
        model_kwargs={"temperature": 0.5, "max_length": 500}
    )

def generate_response(prompt: str, llm, history: List[dict]) -> str:
    """
    Generates a response using the LLM. It prepends the conversation history
    to the current prompt to maintain context (simple memory).
    """
    # 1. Construct the history context string
    history_str = ""
    for message in history:
        if message["role"] == "user":
            history_str += f"User: {message['content']}\n"
        elif message["role"] == "assistant":
            history_str += f"Assistant: {message['content']}\n"
    
    # 2. Construct the full prompt with instructions
    full_prompt = (
        "You are a helpful, friendly, and concise chatbot. Respond to the user's question."
        f"\n\n---CONVERSATION HISTORY---\n{history_str}"
        f"\n\n---NEW MESSAGE---\nUser: {prompt}\nAssistant:"
    )

    # 3. Invoke the LLM
    try:
        response = llm.invoke(full_prompt)
        # Clean up any potential leading/trailing context phrases the model might generate
        return response.strip().replace("Assistant:", "").strip()
    except Exception as e:
        # Catch errors common on free tiers (like rate limiting)
        st.error(f"Error during response generation: {e}")
        return "Sorry, the free service is currently unavailable or busy. Please wait a moment and try a shorter query."

# --- Main Streamlit Application ---

def main():
    st.set_page_config(page_title="Free HF Chatbot", layout="wide")
    st.title("💬 Simple Conversational Chatbot")
    st.caption("Free LLM (Flan-T5 Large) via Streamlit Cloud and Hugging Face.")
    
    # --- Sidebar and Deployment Info ---
    with st.sidebar:
        st.header("Free Deployment Info")
        st.markdown(
            """
            This app uses the **Hugging Face Inference API** to call a free, 
            open-source model. It is hosted for free on **Streamlit Community Cloud**.
            """
        )
        st.markdown("---")
        st.markdown("To deploy this yourself, you only need to set the `HF_TOKEN` secret in Streamlit.")
        st.markdown("LLM: **google/flan-t5-large**")

    # 1. Initialization and LLM Loading
    try:
        hf_token = get_hf_token()
        llm = get_hf_llm(hf_token)
    except Exception:
        # Execution halts within get_hf_token() if key is missing
        return 

    # 2. Initialize Chat History in Session State
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "avatar": "🤖", "content": "Hello! I'm a free, open-source LLM. What can I help you with today?"}
        ]

    # 3. Display Chat Messages from History
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.write(message["content"])

    # 4. Handle User Input
    if user_prompt := st.chat_input("Ask a question..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "avatar": "👤", "content": user_prompt})
        with st.chat_message("user", avatar="👤"):
            st.write(user_prompt)

        # Generate assistant response
        with st.spinner("The open-source model is thinking..."):
            # Pass only the previous history (excluding the current user prompt)
            previous_history = st.session_state.messages[:-1]
            assistant_response = generate_response(user_prompt, llm, previous_history)
        
        # Add assistant message to history and display
        st.session_state.messages.append({"role": "assistant", "avatar": "🤖", "content": assistant_response})
        with st.chat_message("assistant", avatar="🤖"):
            st.write(assistant_response)

if __name__ == "__main__":
    main()

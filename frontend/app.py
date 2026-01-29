import streamlit as st
import requests
import json
import time
import os

# --- Configuration ---
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1/chat/query")
PAGE_TITLE = "Voice RAG System"
PAGE_ICON = "🧠"

# --- Page Setup ---
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp { background: linear-gradient(to bottom right, #0e1117, #1a1c24); color: #ffffff; }
    h1 { background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    div[data-testid="stChatMessage"] { padding: 1.5rem; border-radius: 20px; margin-bottom: 1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
    div[data-testid="stChatMessage"][data-test-user="true"] { background: rgba(58, 123, 213, 0.2); border: 1px solid rgba(58, 123, 213, 0.5); margin-left: 20%; }
    div[data-testid="stChatMessage"][data-test-user="false"] { background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); margin-right: 20%; }
    .stAudioInput { position: fixed; bottom: 40px; left: 50%; transform: translateX(-50%); width: 500px; max-width: 90%; z-index: 1000; background: rgba(26, 28, 36, 0.9); backdrop-filter: blur(10px); padding: 15px; border-radius: 50px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5); border: 1px solid rgba(58, 123, 213, 0.3); }
    section[data-testid="stFileUploaderDropzone"] { min-height: 0px; }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ FIX: Track the last audio we processed to prevent loops
if "last_processed_audio" not in st.session_state:
    st.session_state.last_processed_audio = None

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")
    language = st.selectbox("🗣️ Input Language", ["hi-IN", "en-IN", "bn-IN", "ta-IN", "te-IN"], index=0)
    if st.button("🗑️ Clear Chat", type="primary"):
        st.session_state.messages = []
        st.session_state.last_processed_audio = None # Reset audio state
        st.rerun()

# --- Main Interface ---
st.markdown(f"# {PAGE_TITLE}")
st.caption("🤖 AI Voice Assistant • Powered by RAG")

# --- Chat History ---
if not st.session_state.messages:
    st.markdown("""<div style="text-align: center; padding: 50px; opacity: 0.6;"><h3>👋 Hello!</h3><p>Tap the microphone below and ask me anything.</p></div>""", unsafe_allow_html=True)
    
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.write(msg["content"])
            with st.expander("📝 Original Audio"):
                st.code(msg["metadata"]["original_transcription"], language="text")
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.write(msg["content"])
            # In app.py, inside the message loop:

            # --- ✨ IMPROVED: Sources with Links ---
            if "metadata" in msg:
                meta = msg["metadata"]
                with st.expander("📚 Sources & Wikipedia Links"):
                    st.caption(f"⏱️ Processed in {meta.get('processing_time',0)}s")
                    
                    # 1. robust Parsing Logic
                    raw_articles = meta.get('article', '[]')
                    topics = []
                    
                    try:
                        # Case A: It's a string representation of a list "['Topic A', 'Topic B']"
                        if raw_articles.strip().startswith("["):
                            import ast
                            topics = ast.literal_eval(raw_articles)
                        # Case B: It's just a single string "Topic A"
                        else:
                            topics = [raw_articles]
                    except:
                        # Fallback: Just use the raw string
                        topics = [raw_articles]
                    
                    # 2. Render Links
                    st.markdown("**References:**")
                    
                    # Filter out empty or None
                    valid_topics = [t for t in topics if t and isinstance(t, str)]
                    
                    if not valid_topics:
                        st.caption("No specific articles found.")
                    else:
                        for topic in valid_topics:
                            # Heuristic: Create a Wikipedia URL
                            # 1. Trim whitespace
                            # 2. Replace spaces with underscores
                            clean_topic = topic.strip().replace(" ", "_")
                            url = f"https://en.wikipedia.org/wiki/{clean_topic}"
                            
                            # Render clickable link
                            st.markdown(f"🔗 [{topic}]({url})")
# --- Audio Input ---
audio_value = st.audio_input("Tap to speak...")

# --- ✅ FIX: THE GATEKEEPER ---
# Only process if audio exists AND it is different from the last one we processed
if audio_value and audio_value != st.session_state.last_processed_audio:
    
    with st.container():
        with st.status("🧠 Processing...", expanded=True) as status:
            try:
                # 1. Prepare Chat History (New Step!)
                # We convert the Streamlit message history into a clean format for the API
                history_payload = []
                for msg in st.session_state.messages:
                    history_payload.append({
                        "role": "user" if msg["role"] == "user" else "model",
                        "content": msg["content"]
                    })

                # 2. Prepare Data Payload
                files = {"audio_file": ("input.wav", audio_value, "audio/wav")}
                data = {
                    "language_code": language,
                    "chat_history": json.dumps(history_payload)  # ✅ Sending History to Backend
                }
                
                status.write("🎧 Transcribing & Contextualizing...")
                response = requests.post(API_URL, files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    status.write("📚 Searching Knowledge Base...")
                    time.sleep(0.5) 
                    
                    status.update(label="✅ Ready!", state="complete", expanded=False)
                    
                    # Add User's "Translated" Message
                    st.session_state.messages.append({
                        "role": "user",
                        "content": result.get("translated_text"),
                        "metadata": {"original_transcription": result.get("transcribed_text")}
                    })
                    
                    # Add Bot's Answer
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result.get("llm_answer"),
                        "metadata": {
                            "processing_time": result.get("processing_time_seconds"),
                            "article": result.get("wikipedia_article"),
                            "refined_query": result.get("extracted_keyword") # Useful to see how it rewrote your query
                        }
                    })
                    
                    # ✅ FIX: Mark this audio as processed to prevent loops
                    st.session_state.last_processed_audio = audio_value
                    
                    st.rerun()
                else:
                    st.error(f"Error: {response.text}")
                    status.update(label="❌ Failed", state="error")
                    
            except Exception as e:
                st.error(f"Connection Error: {e}")

st.write("<br><br><br><br><br><br><br>", unsafe_allow_html=True)
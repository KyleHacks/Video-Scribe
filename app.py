import streamlit as st
import openai
import tempfile
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Admin token and backend API key from environment
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
BACKEND_OPENAI_KEY = os.getenv("OPENAI_API_KEY")

st.title("🎧 Multilingual MP4 Transcription App")
st.write("Upload an MP4 file and get the transcription using OpenAI Whisper.")

# Input box for user to provide their OpenAI API key or admin token
user_key_input = st.text_input("🔑 Enter your OpenAI API Key:", type="password")

# Determine which API key to use
if user_key_input.startswith("#"):
    # Use backend key only if token matches
    if user_key_input[1:] == ADMIN_TOKEN:
        openai.api_key = BACKEND_OPENAI_KEY
        st.success("✅ Admin token accepted. Using backend API key.")
    else:
        st.error("❌ Invalid admin token.")
        st.stop()
else:
    openai.api_key = user_key_input

# Upload section
uploaded_file = st.file_uploader("📤 Upload MP4 File", type=["mp4"])

if uploaded_file is not None:
    st.video(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.info("Transcribing... Please wait ⏳")

    try:
        # Whisper transcription
        with open(tmp_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)

        st.success("✅ Transcription Completed!")
        st.text_area("📜 Transcription Output", transcript["text"], height=300)

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

    # Clean up
    os.remove(tmp_path)

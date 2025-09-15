import streamlit as st
import openai
import tempfile
import os
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.silence import split_on_silence
import math

# Load .env file
load_dotenv()

# Admin token and backend API key from environment
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
BACKEND_OPENAI_KEY = os.getenv("OPENAI_API_KEY")

def remove_silence_from_audio(audio_path, output_path):
    """Remove silence from audio file and save as MP3"""
    try:
        # Load audio file
        audio = AudioSegment.from_file(audio_path)
        
        # Split on silence (silence threshold: -40dBFS, min silence length: 500ms)
        chunks = split_on_silence(
            audio,
            min_silence_len=500,  # 500ms of silence
            silence_thresh=-40,   # -40dBFS threshold
            keep_silence=100      # Keep 100ms of silence at edges
        )
        
        # Concatenate all non-silent chunks
        condensed_audio = AudioSegment.empty()
        for chunk in chunks:
            condensed_audio += chunk
        
        # Export as MP3
        condensed_audio.export(output_path, format="mp3")
        return output_path
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return audio_path

def segment_audio(audio_path, segment_duration_minutes):
    """Split audio into segments of specified duration"""
    try:
        audio = AudioSegment.from_file(audio_path)
        segment_length_ms = segment_duration_minutes * 60 * 1000  # Convert to milliseconds
        
        segments = []
        total_duration = len(audio)
        
        for i in range(0, total_duration, segment_length_ms):
            segment = audio[i:i + segment_length_ms]
            
            # Create temporary file for segment
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                segment.export(tmp.name, format="mp3")
                segments.append({
                    'path': tmp.name,
                    'start_time': i // 1000,  # Convert to seconds
                    'end_time': min((i + segment_length_ms) // 1000, total_duration // 1000)
                })
        
        return segments
    except Exception as e:
        st.error(f"Error segmenting audio: {e}")
        return []

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

# Settings section
st.subheader("⚙️ Transcription Settings")

col1, col2 = st.columns(2)

with col1:
    condensed_audio = st.checkbox("🎵 Condensed Audio",
                                 help="Remove silence from audio to reduce file size and processing time")

with col2:
    enable_segmentation = st.checkbox("✂️ Enable Segmentation",
                                     help="Process large files in smaller segments")

if enable_segmentation:
    segment_duration = st.slider("Segment Duration (minutes)",
                                min_value=1, max_value=10, value=2,
                                help="Duration of each segment in minutes")
else:
    segment_duration = None

# Upload section
uploaded_file = st.file_uploader("📤 Upload MP4 File", type=["mp4"])

if uploaded_file is not None:
    st.video(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Process audio based on settings
    processed_audio_path = tmp_path
    
    if condensed_audio:
        st.info("🎵 Processing audio to remove silence...")
        condensed_path = tmp_path.replace('.mp4', '_condensed.mp3')
        processed_audio_path = remove_silence_from_audio(tmp_path, condensed_path)
        st.success("✅ Silence removed from audio!")

    # Handle segmentation or regular transcription
    if enable_segmentation and segment_duration:
        st.info("✂️ Segmenting audio for processing...")
        segments = segment_audio(processed_audio_path, segment_duration)
        
        if segments:
            st.success(f"✅ Audio split into {len(segments)} segments!")
            
            # Initialize progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            full_transcript = ""
            
            # Process each segment
            for i, segment in enumerate(segments):
                status_text.text(f"Transcribing segment {i+1}/{len(segments)}...")
                progress_bar.progress((i) / len(segments))
                
                try:
                    with open(segment['path'], "rb") as audio_file:
                        transcript = openai.Audio.transcribe("whisper-1", audio_file)
                    
                    # Add timestamp and transcript
                    start_min = segment['start_time'] // 60
                    start_sec = segment['start_time'] % 60
                    end_min = segment['end_time'] // 60
                    end_sec = segment['end_time'] % 60
                    
                    segment_header = f"\n[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]\n"
                    full_transcript += segment_header + transcript["text"] + "\n"
                    
                    # Clean up segment file
                    os.remove(segment['path'])
                    
                except Exception as e:
                    st.error(f"❌ Error transcribing segment {i+1}: {e}")
                    continue
            
            # Complete progress
            progress_bar.progress(1.0)
            status_text.text("✅ All segments transcribed!")
            
            st.success("✅ Segmented Transcription Completed!")
            st.text_area("📜 Full Transcription Output", full_transcript, height=400)
        else:
            st.error("❌ Failed to segment audio file")
    
    else:
        # Regular transcription (non-segmented)
        st.info("Transcribing... Please wait ⏳")
        
        try:
            with open(processed_audio_path, "rb") as audio_file:
                transcript = openai.Audio.transcribe("whisper-1", audio_file)

            st.success("✅ Transcription Completed!")
            st.text_area("📜 Transcription Output", transcript["text"], height=300)

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

    # Clean up temporary files
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    if condensed_audio and processed_audio_path != tmp_path and os.path.exists(processed_audio_path):
        os.remove(processed_audio_path)

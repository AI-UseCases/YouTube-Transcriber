import random
import streamlit as st
import io
import os

from transformers import pipeline
import torch
import yt_dlp
from silero_vad import load_silero_vad, get_speech_timestamps
import numpy as np
import pydub
from litellm import completion

# --- Model Loading and Caching ---
@st.cache_resource
def load_transcriber(_device):
    """Loads the Whisper transcription model."""
    transcriber = pipeline(model="openai/whisper-large-v3-turbo", device=_device)
    return transcriber

@st.cache_resource
def load_vad_model():
    """Loads the Silero VAD model."""
    return load_silero_vad()

# --- Audio Processing Functions ---
@st.cache_resource
def download_and_convert_audio(video_url, audio_format="wav"):
    """Downloads and converts audio from a YouTube video.

    Args:
        video_url (str): The URL of the YouTube video.
        audio_format (str): The desired audio format (e.g., "wav", "mp3").

    Returns:
        tuple: (audio_bytes, audio_format, info_dict) or (None, None, None) on error.
    """
    status_message = st.empty()
    status_message.text("Downloading audio...")
    try:
        ydl_opts = {
            'format': f'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': audio_format,
            }],
            'outtmpl': '%(id)s.%(ext)s',
            'noplaylist': True,
            'progress_hooks': [lambda d: update_download_progress(d, status_message)],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            if 'entries' in info:
                info = info['entries'][0]
            video_id = info['id']
            filename = f"{video_id}.{audio_format}"

            audio_formats = [f for f in info.get('formats', []) if f.get('acodec') != 'none' and f.get('vcodec') == 'none']
            if not audio_formats:
                st.warning(f"No audio-only format found. Downloading and converting from best video format to {audio_format}.")
                ydl_opts['format'] = 'best'

            ydl.download([video_url])
            status_message.text(f"Audio downloaded and converted to {audio_format}.")

            with open(filename, 'rb') as audio_file:
                audio_bytes = audio_file.read()

            os.remove(filename)
            return audio_bytes, audio_format, info
    except Exception as e:
        st.error(f"Error during download or conversion: {e}")
        return None, None, None

def update_download_progress(d, status_message):
    """Updates the download progress in the Streamlit UI."""
    if d['status'] == 'downloading':
        p = round(d['downloaded_bytes'] / d['total_bytes'] * 100)
        status_message.text(f"Downloading: {p}%")

@st.cache_data
def split_audio_by_vad(audio_data: bytes, ext: str, _vad_model, sensitivity: float, max_duration: int = 30, return_seconds: bool = True):
    """Splits audio into chunks based on voice activity detection (VAD).

    Args:
        audio_data (bytes): The audio data as bytes.
        ext (str): The audio file extension.
        _vad_model: The VAD model.
        sensitivity (float): The VAD sensitivity (0.0 to 1.0).
        max_duration (int): The maximum duration of each chunk in seconds.
        return_seconds (bool): Whether to return timestamps in seconds.

    Returns:
        list: A list of dictionaries, where each dictionary represents an audio chunk.
              Returns an empty list if no speech segments are detected or an error occurs.
    """
    
    if not audio_data:
        st.error("No audio data received.")
        return []

    try:
        audio = pydub.AudioSegment.from_file(io.BytesIO(audio_data), format=ext)
        rate = audio.frame_rate
        
        # Convert to mono if stereo for compatibility with VAD
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # Calculate dynamic VAD parameters based on sensitivity
        window_size_samples = int(512 + (1536 - 512) * (1 - sensitivity))
        speech_threshold = 0.5 + (0.95 - 0.5) * sensitivity
        
        samples = np.array(audio.get_array_of_samples())

        speech_timestamps = get_speech_timestamps(
            samples,
            _vad_model,
            sampling_rate=rate,
            return_seconds=return_seconds,
            window_size_samples=window_size_samples,
            threshold=speech_threshold,
        )

        if not speech_timestamps:
            st.warning("No speech segments detected.")
            return []

        speech_timestamps[0]["start"] = 0.
        speech_timestamps[-1]['end'] = audio.duration_seconds
        for i, chunk in enumerate(speech_timestamps[1:], start=1):
            chunk["start"] = speech_timestamps[i - 1]['end']

        aggregated_segments = []
        if speech_timestamps:
            current_segment_start = speech_timestamps[0]['start']
            current_segment_end = speech_timestamps[0]['end']
            for segment in speech_timestamps[1:]:
                if segment['start'] - current_segment_start >= max_duration:
                    aggregated_segments.append({'start': current_segment_start, 'end': current_segment_end})
                    current_segment_start = segment['start']
                    current_segment_end = segment['end']
                else:
                    current_segment_end = segment['end']
            aggregated_segments.append({'start': current_segment_start, 'end': current_segment_end})
        
        if not aggregated_segments:
            return []

        chunks = []
        for segment in aggregated_segments:
            start_ms = int(segment['start'] * 1000)
            end_ms = int(segment['end'] * 1000)
            chunk = audio[start_ms:end_ms]
            chunk_io = io.BytesIO()
            chunk.export(chunk_io, format=ext)
            chunks.append({
                'data': chunk_io.getvalue(),
                'start': segment['start'],
                'end': segment['end']
            })
            chunk_io.close()
        return chunks
    except Exception as e:
        st.error(f"Error processing audio in split_audio_by_vad: {str(e)}")
        return []
    finally:
        if 'audio' in locals():
            del audio
        if 'samples' in locals():
            del samples

@st.cache_data
def transcribe_batch(batch, _transcriber, language=None):
    """Transcribes a batch of audio chunks.

    Args:
        batch (list): A list of audio chunk dictionaries.
        _transcriber: The transcription model.
        language (str, optional): The language of the audio (e.g., "en", "es"). Defaults to None (auto-detection).

    Returns:
        list: A list of dictionaries, each containing the transcription, start, and end time of a chunk.
              Returns an empty list if an error occurs.
    """
    transcriptions = []
    for i, chunk_data in enumerate(batch):
        try:
            generate_kwargs = {
                "task": "transcribe",
                "return_timestamps": True,
                "language": language
            }

            transcription = _transcriber(
                chunk_data['data'],
                generate_kwargs=generate_kwargs
            )
            transcriptions.append({
                'text': transcription["text"],
                'start': chunk_data['start'],
                'end': chunk_data['end']}
            )
        except Exception as e:
            st.error(f"Error transcribing chunk {i}: {str(e)}")
            return []
    return transcriptions

# --- Streamlit App ---
def setup_ui():
    """Sets up the Streamlit user interface."""
    st.title("YouTube Video Transcriber")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        transcribe_option = st.checkbox("Transcribe", value=True)
    with col2:
        download_audio_option = st.checkbox("Download Audio", value=False)
    with col3:
        download_video_option = st.checkbox("Download Video", value=False)
    with col4:
        pass

    video_url = st.text_input("YouTube Video Link:", key="video_url")
    language = st.text_input("Language (two-letter code, e.g., 'en', 'es', leave empty for auto-detection):", max_chars=2, key="language")
    batch_size = st.number_input("Batch Size", min_value=1, value=2, key="batch_size")
    vad_sensitivity = st.slider("VAD Sensitivity", min_value=0.0, max_value=1.0, value=0.1, step=0.05, key="vad_sensitivity")
    
    # Use session state to manage audio format selection and reset
    if 'reset_audio_format' not in st.session_state:
        st.session_state.reset_audio_format = False

    if 'audio_format' not in st.session_state or st.session_state.reset_audio_format:
        st.session_state.audio_format = "wav"  # Default value
        st.session_state.reset_audio_format = False

    audio_format = st.selectbox("Audio Format", ["wav", "mp3", "ogg", "flac"], key="audio_format_widget", index=["wav", "mp3", "ogg", "flac"].index(st.session_state.audio_format))
    st.session_state.audio_format = audio_format
    
    if download_video_option:
        video_format = st.selectbox("Video Format", ["mp4", "webm"], index=0, key="video_format")
    else:
        video_format = "mp4"

    process_button = st.button("Process")

    return video_url, language, batch_size, transcribe_option, download_audio_option, download_video_option, process_button, vad_sensitivity, audio_format, video_format

@st.cache_resource
def initialize_models():
    """Initializes the transcription and VAD models."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transcriber = load_transcriber(device)
    vad_model = load_vad_model()
    return transcriber, vad_model

def process_transcription(video_url, vad_sensitivity, batch_size, transcriber, vad_model, audio_format, language=None):
    """Downloads, processes, and transcribes the audio from a YouTube video.

    Args:
        video_url (str): The URL of the YouTube video.
        vad_sensitivity (float): The VAD sensitivity.
        batch_size (int): The batch size for transcription.
        transcriber: The transcription model.
        vad_model: The VAD model.
        language (str, optional): The language of the audio. Defaults to None.

    Returns:
        tuple: (full_transcription, audio_data, audio_format, info) or (None, None, None, None) on error.
    """
    audio_data, audio_format, info = download_and_convert_audio(video_url, audio_format)
    if not audio_data:
        return None, None, None, None

    chunks = split_audio_by_vad(audio_data, audio_format, vad_model, vad_sensitivity)
    if not chunks:
        return None, None, None, None

    total_chunks = len(chunks)
    transcriptions = []
    progress_bar = st.progress(0)
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        batch_transcriptions = transcribe_batch(batch, transcriber, language)
        transcriptions.extend(batch_transcriptions)
        progress_bar.progress((i + len(batch)) / total_chunks)

    progress_bar.empty()
    st.success("Transcription complete!")

    full_transcription = ""
    for chunk in transcriptions:
        start_time = format_seconds(chunk['start'])
        end_time = format_seconds(chunk['end'])
        full_transcription += f"[{start_time} - {end_time}]: {chunk['text'].strip()}\n\n"
    formatted_transcription = format_transcript(full_transcription)

    return full_transcription, formatted_transcription, audio_data, audio_format, info

def format_seconds(seconds):
    """Formats seconds into HH:MM:SS string."""
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def download_video(video_url, video_format):
    """Downloads video from YouTube using yt-dlp."""
    status_message = st.empty()
    status_message.text("Downloading video...")
    try:
        ydl_opts = {
            'format': f'bestvideo[ext={video_format}]+bestaudio[ext=m4a]/best[ext={video_format}]/best',
            'outtmpl': '%(title)s.%(ext)s',
            'noplaylist': True,
            'progress_hooks': [lambda d: update_download_progress(d, status_message)],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            video_filename = ydl.prepare_filename(info_dict)
            video_title = info_dict.get("title", "video")
            status_message.text(f"Video downloaded: {video_title}")

            with open(video_filename, 'rb') as video_file:
                video_bytes = video_file.read()

            os.remove(video_filename)

            return video_bytes, video_filename, info_dict
    except Exception as e:
        st.error(f"Error during video download: {e}")
        return None, None, None

def format_transcript(input_transcription):
    

    # os.environ["GEMINI_API_KEY"] = "..."

    sys_prompt = """
    Video Transcription Formatting

    As an LLM formatting provided video transcriptions (in any language), transform spoken language into clear, readable text. Prioritize readability, consistency, and context, adapting to the specific language conventions. **Do not hallucinate or add any information not present in the original transcript.**

    *   **Sentences:** Restructure long, rambling sentences; correct grammatical errors *while preserving the original meaning*; use proper punctuation appropriate for the language.
    *   **Reading:** Italicize/quote read text; clearly separate from explanations.
    *   **Repetitions:** Remove unnecessary repetitions unless for emphasis.
    """.strip()
    messages = [{"content": sys_prompt, "role": "system"},
                 {"content": f"Format the following video transcription: {input_transcription}", "role": "user"}]

    response = completion(model="gemini/gemini-2.0-flash-exp", messages=messages)
    formatted_text = response.choices[0].message.content
    return formatted_text

def main():
    """Main function to run the Streamlit application."""

    # Initialize session state variables
    if 'full_transcription' not in st.session_state:
        st.session_state.full_transcription = None
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
    if 'info' not in st.session_state:
        st.session_state.info = None
    if 'video_data' not in st.session_state:
        st.session_state.video_data = None
    if 'video_filename' not in st.session_state:
        st.session_state.video_filename = None

    transcriber, vad_model = initialize_models()
    
    # Call setup_ui() to get UI element values
    video_url, language, batch_size, transcribe_option, download_audio_option, download_video_option, process_button, vad_sensitivity, audio_format, video_format = setup_ui()

    # transcription_output = st.empty()
    if st.session_state.full_transcription:
        st.text_area("Transcription:", value=st.session_state.full_transcription, height=300, key=random.random())

    if process_button:
        st.session_state.full_transcription = None
        st.session_state.audio_data = None
        st.session_state.info = None
        st.session_state.video_data = None
        st.session_state.video_filename = None
        st.session_state.reset_audio_format = True

        if not video_url:
            st.error("Please enter a YouTube video link.")
            return

        if transcribe_option:
            st.session_state.full_transcription, st.session_state.formatted_transcription, st.session_state.audio_data, st.session_state.audio_format, st.session_state.info = process_transcription(video_url, vad_sensitivity, batch_size, transcriber, vad_model, audio_format, language)
            if st.session_state.full_transcription:
                st.text_area("Transcription:", value=st.session_state.full_transcription, height=300, key=random.random())
            if st.session_state.formatted_transcription:
                st.text_area("Formatted Transcription:", value=st.session_state.formatted_transcription, height=300, key=random.random())


        if download_audio_option:
            if st.session_state.audio_data is None or st.session_state.audio_format is None or st.session_state.info is None:
                st.session_state.audio_data, st.session_state.audio_format, st.session_state.info = download_and_convert_audio(video_url, audio_format)

        if download_video_option:
            st.session_state.video_data, st.session_state.video_filename, st.session_state.info = download_video(video_url, video_format)

    # Download button logic (moved after setup_ui() call)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.full_transcription and transcribe_option:
            st.download_button(
                label="Download Transcription (TXT)",
                data=st.session_state.full_transcription,
                file_name=f"{st.session_state.info['id']}_transcription.txt",
                mime="text/plain"
            )

    with col2:
        # Now download_audio_option is defined
        if st.session_state.audio_data is not None and download_audio_option:
            st.download_button(
                label=f"Download Audio ({st.session_state.audio_format})",
                data=st.session_state.audio_data,
                file_name=f"{st.session_state.info['id']}.{st.session_state.audio_format}",
                mime=f"audio/{st.session_state.audio_format}"
            )

    with col3:
        if st.session_state.video_data is not None and download_video_option:
            st.download_button(
                label="Download Video",
                data=st.session_state.video_data,
                file_name=st.session_state.video_filename,
                mime=f"video/{video_format}"
            )

if __name__ == "__main__":
    main()

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

VAD_SENSITIVITY = 0.1

# --- Model Loading and Caching ---
@st.cache_resource
def load_transcriber(_device):
    transcriber = pipeline(model="openai/whisper-large-v3-turbo", device=_device)
    return transcriber

@st.cache_resource
def load_vad_model():
    return load_silero_vad()

# --- Audio Processing Functions ---
@st.cache_resource
def download_and_convert_audio(video_url):
    status_message = st.empty()
    status_message.text("Downloading audio...")
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': '%(id)s.%(ext)s',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            video_id = info['id']
            filename = f"{video_id}.wav"
            ydl.download([video_url])
            status_message.text("Audio downloaded and converted.")
            
            # Read the file and return its contents
            with open(filename, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            
            # Clean up the temporary file
            os.remove(filename)
            
            return audio_bytes, 'wav'
    except Exception as e:
        st.error(f"Error during download or conversion: {e}")
        return None, None

def aggregate_speech_segments(speech_timestamps, max_duration=30):
    """Aggregates speech segments into chunks with a maximum duration,
    merging the last segment if it's contained within the second-to-last.

    Args:
        speech_timestamps: A list of dictionaries, where each dictionary represents
                           a speech segment with 'start' and 'end' timestamps
                           (in seconds).
        max_duration: The maximum desired duration of each aggregated segment
                      (in seconds). Defaults to 30.

    Returns:
        A list of dictionaries, where each dictionary represents an aggregated
        speech segment with 'start' and 'end' timestamps.
    """

    if not speech_timestamps:
        return []

    aggregated_segments = []
    current_segment_start = speech_timestamps[0]['start']
    current_segment_end = speech_timestamps[0]['end']

    for segment in speech_timestamps[1:]:
        if segment['start'] - current_segment_start >= max_duration:
            # Start a new segment if the current duration exceeds max_duration
            aggregated_segments.append({'start': current_segment_start, 'end': current_segment_end})
            current_segment_start = segment['start']
            current_segment_end = segment['end']
        else:
            # Extend the current segment
            current_segment_end = segment['end']

    # Add the last segment, checking for redundancy
    last_segment = {'start': current_segment_start, 'end': current_segment_end}
    if aggregated_segments:
        second_last_segment = aggregated_segments[-1]
        if last_segment['start'] >= second_last_segment['start'] and last_segment['end'] <= second_last_segment['end']:
            # Last segment is fully contained in the second-to-last, so don't add it
            pass
        else:
            aggregated_segments.append(last_segment)
    else:
        # If aggregated_segments is empty, add the last segment
        aggregated_segments.append(last_segment)

    return aggregated_segments

@st.cache_data
def split_audio_by_vad(audio_data: bytes, ext: str, _vad_model, sensitivity: float, return_seconds: bool = True):
    if not audio_data:
        st.error("No audio data received.")
        return []
        
    try:
        audio = pydub.AudioSegment.from_file(io.BytesIO(audio_data), format=ext)
        
        # VAD parameters
        rate = audio.frame_rate
        window_size_samples = int(512 + (1536 - 512) * (1 - sensitivity))
        speech_threshold = 0.5 + (0.95 - 0.5) * sensitivity
        
        # Convert audio to numpy array for VAD
        samples = np.array(audio.get_array_of_samples())

        # Get speech timestamps
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

        # rectify timestamps
        speech_timestamps[0]["start"] = 0.
        speech_timestamps[-1]['end'] = audio.duration_seconds
        for i, chunk in enumerate(speech_timestamps[1:], start=1):
            chunk["start"] = speech_timestamps[i-1]['end']

        # Aggregate segments into ~30 second chunks
        aggregated_segments = aggregate_speech_segments(speech_timestamps, max_duration=30)

        if not aggregated_segments:
            return []

        # Create audio chunks based on timestamps
        chunks = []
        for segment in aggregated_segments:
            start_ms = int(segment['start'] * 1000)
            end_ms = int(segment['end'] * 1000)
            chunk = audio[start_ms:end_ms]
            
            # Export chunk to bytes
            chunk_io = io.BytesIO()
            chunk.export(chunk_io, format=ext)
            chunk_data = chunk_io.getvalue()  # Get bytes directly
            
            chunks.append({
                'data': chunk_data,
                'start': segment['start'],
                'end': segment['end']
            })
            chunk_io.close() # Close the BytesIO object after getting the value
        
        return chunks
    except Exception as e:
        st.error(f"Error processing audio in split_audio_by_vad: {str(e)}")
        return []
    finally:
        # Explicitly release pydub resources to prevent memory issues
        if 'audio' in locals():
            del audio
        if 'samples' in locals():
            del samples

@st.cache_data
def transcribe_batch(batch, _transcriber, language=None):
    transcriptions = []
    for i, chunk_data in enumerate(batch):
        try:
            generate_kwargs = {
                "task": "transcribe",
                "return_timestamps": True
            }
            if language:
                generate_kwargs["language"] = language
            
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
    st.title("YouTube Video Transcriber")
    video_url = st.text_input("YouTube Video Link:")
    language = st.text_input("Language (two-letter code, e.g., 'en', 'es', leave empty for auto-detection):", max_chars=2)
    batch_size = st.number_input("Batch Size", min_value=1, max_value=10, value=2)  # Batch size selection
    transcribe_button = st.button("Transcribe")
    return video_url, language,batch_size, transcribe_button

@st.cache_resource
def initialize_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transcriber = load_transcriber(device)
    vad_model = load_vad_model()
    return transcriber, vad_model

def process_transcription(video_url, vad_sensitivity, batch_size, transcriber, vad_model, language=None):
    transcription_output = st.empty()
    audio_data, ext = download_and_convert_audio(video_url)
    if not audio_data:
        return
    
    chunks = split_audio_by_vad(audio_data, ext, vad_model, vad_sensitivity)
    if not chunks:
        return

    total_chunks = len(chunks)
    transcriptions = []
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        batch_transcriptions = transcribe_batch(batch, transcriber, language)
        transcriptions.extend(batch_transcriptions)
        display_transcription(transcriptions, transcription_output)

    st.success("Transcription complete!")

def display_transcription(transcriptions, output_area):
    full_transcription = ""
    for chunk in transcriptions:
        start_time = format_seconds(chunk['start'])
        end_time = format_seconds(chunk['end'])
        full_transcription += f"[{start_time} - {end_time}]: {chunk['text'].strip()}\n\n"
    output_area.text_area("Transcription:", value=full_transcription, height=300, key=random.random())

def format_seconds(seconds):
    """Formats seconds into HH:MM:SS string."""
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def main():
    transcriber, vad_model = initialize_models()
    video_url, language, batch_size, transcribe_button = setup_ui()
    if transcribe_button:
        if not video_url:
            st.error("Please enter a YouTube video link.")
            return
        process_transcription(video_url, VAD_SENSITIVITY, batch_size, transcriber, vad_model, language)

if __name__ == "__main__":
    main()

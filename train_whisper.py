import os
import torch
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
# import whisper  <-- Removed to fix numpy version conflict with numba
from transformers import pipeline
import re
from datetime import datetime

# --- Configuration ---
WHISPER_MODEL_ID = "openai/whisper-large-v3"
AUDIO_EXTENSIONS = ('.wav', '.mp3', '.m4a', '.flac', '.mp4')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def check_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📡 Using Device: {device.upper()}")
    return device

def find_audio_file(directory):
    """Finds the first audio file in the given directory."""
    for file in os.listdir(directory):
        if file.lower().endswith(AUDIO_EXTENSIONS):
            return os.path.join(directory, file)
    return None

def preprocess_audio(input_path):
    """Standardize and reduce noise in audio."""
    print(f"🛠️  Preprocessing: {os.path.basename(input_path)}")
    
    # Load audio
    audio, sr = librosa.load(input_path, sr=16000, mono=True)
    
    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    # Noise Reduction
    print("🧹 Reducing background noise...")
    reduced_audio = nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.75)
    
    # Save processed audio
    processed_path = os.path.join(OUTPUT_DIR, "processed_audio.wav")
    sf.write(processed_path, reduced_audio, sr)
    
    return processed_path

def transcribe_whisper(audio_path, device):
    """Transcription using the Whisper model."""
    print(f"🎙️  Starting Whisper Transcription ({WHISPER_MODEL_ID})...")
    
    # Initialize pipeline
    device_idx = 0 if device == "cuda" else -1
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL_ID,
        torch_dtype=torch_dtype,
        device=device_idx,
        chunk_length_s=30,
        model_kwargs={"low_cpu_mem_usage": True},
    )
    
    # Run inference
    result = pipe(audio_path, return_timestamps=False)
    text = result["text"]
    
    # Cleanup post-processing
    text = clean_transcription(text)
    
    return text

def clean_transcription(text):
    """Clean up common Whisper artifacts and hallucinations."""
    # Remove excessive 'Thank you' repetitions (common in silent chunks)
    text = re.sub(r'(Thank you\s*){3,}', 'Thank you. ', text)
    # Remove repetitive digit hallucinations
    text = re.sub(r'(\s\d\.?){4,}', '', text)
    return text.strip()

def save_transcript(text, original_filename):
    """Saves the transcription to a text file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(original_filename))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_transcript_{timestamp}.txt")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"✅ Transcription saved to: {output_path}")
    return output_path

def main():
    print("🚀 --- Whisper Transcription Pipeline --- 🚀")
    device = check_device()
    
    # Search for audio in the current folder
    audio_file = find_audio_file(OUTPUT_DIR)
    
    if not audio_file:
        print(f"❌ No audio files found in {OUTPUT_DIR}")
        print(f"Please place a .wav, .mp3, or .m4a file in this folder.")
        return

    print(f"📂 Found Audio: {audio_file}")
    
    try:
        # Step 1: Preprocess
        processed_audio = preprocess_audio(audio_file)
        
        # Step 2: Transcribe
        transcript = transcribe_whisper(processed_audio, device)
        
        # Step 3: Display Preview
        print("\n--- TRANSCRIPTION PREVIEW ---")
        preview = transcript[:500] + "..." if len(transcript) > 500 else transcript
        print(preview)
        print("-----------------------------\n")
        
        # Step 4: Save
        save_transcript(transcript, audio_file)
        
    except Exception as e:
        print(f"💥 Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

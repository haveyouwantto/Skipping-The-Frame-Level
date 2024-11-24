from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from io import BytesIO
import os
import uvicorn
import numpy as np
import tempfile
import yt_dlp as youtube_dl
from infer import TranskunInfer
import librosa
import uuid

app = FastAPI()

# Initialize the transcription model
model = TranskunInfer(device="cuda" if torch.cuda.is_available() else "cpu")

def read_audio(path, normalize=True):
    y, sr = librosa.load(path)
    y = y.reshape(-1, 1)
    if normalize:
        y = np.float32(y) / 2**15
    return sr, y

class TranscriptionResult(BaseModel):
    status: str
    reason: str = None


def download_audio(url: str) -> str:
    """
    Download audio from a URL using yt-dlp's Python API, save it to a file, 
    and return the file path.

    Args:
        url (str): URL to download the audio.

    Returns:
        str: Path to the downloaded audio file.
    """
    # Create a temporary directory for storing the downloaded audio
    temp_dir = tempfile.mkdtemp()
    
    # Define the audio file path
    audio_path = os.path.join(temp_dir, "%s.ogg" % (uuid.uuid4().hex))  # Use .ogg format as specified

    # yt-dlp options for extracting audio
    ydl_opts = {
        'format': 'bestaudio/best',   # Download best quality audio
        'extractaudio': True,         # Extract audio only
        'audioformat': 'wav',         # Specify audio format (ogg)
        'outtmpl': audio_path,        # Specify output file path
        'overwrites': True,           # Overwrite existing files
        'quiet': True,                # Suppress yt-dlp output
    }

    try:
        # Download audio using yt-dlp
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])  # Download the audio

        # Return the path to the downloaded audio file
        return audio_path
    except Exception as e:
        raise RuntimeError(f"Failed to download audio: {str(e)}")

@app.post("/v1/transcribe", response_model=TranscriptionResult)
async def transcribe(file: UploadFile = None, url: str = Form(None)):
    """
    Transcribe an audio file or URL to MIDI.

    Args:
        file (UploadFile): Uploaded audio file.
        url (str): URL to download audio if file is not provided.

    Returns:
        StreamingResponse: MIDI file if successful.
        JSONResponse: Error message if failure.
    """
    try:
        if not file and not url:
            return JSONResponse(status_code=400, content={"status": "error", "reason": "No file or URL provided"})

        # Handle audio input
        if file:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            with temp_file as f:
                f.write(await file.read())
            audio_path = temp_file.name
        else:
            audio_path = download_audio(url)

        # Load audio and convert to MIDI
        sr, audio = read_audio(audio_path)
        midi_bytes = model.get_midi(audio=audio, fs=sr)

        # Clean up temporary files
        os.remove(audio_path)

        # Return MIDI file
        return StreamingResponse(BytesIO(midi_bytes), media_type="audio/midi")

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "reason": str(e)})

    finally:
        if file and os.path.exists(audio_path):
            os.remove(audio_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

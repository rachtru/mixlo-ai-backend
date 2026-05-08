"""
Mixlo AI — FastAPI Backend
Accepts stem uploads, classifies them with LibROSA,
mixes with Pedalboard, exports a final WAV.
"""

import io
import os
import uuid
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from mixer import classify_stem, mix_stems

app = FastAPI(title="Mixlo AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "mixlo_uploads"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "mixlo_output"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok", "service": "Mixlo AI"}


@app.post("/analyze")
async def analyze_stems(files: list[UploadFile] = File(...)):
    """
    Step 1: Upload stems and get back classification results.
    Does NOT mix yet — lets the user confirm labels before mixing.
    """
    if not files:
        raise HTTPException(400, "No files uploaded.")
    if len(files) > 16:
        raise HTTPException(400, "Maximum 16 stems per mix.")

    results = []
    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir()

    for upload in files:
        if not upload.filename:
            continue
        ext = Path(upload.filename).suffix.lower()
        if ext not in {".wav", ".mp3", ".aiff", ".flac"}:
            raise HTTPException(400, f"{upload.filename}: unsupported format (use WAV/MP3/AIFF/FLAC).")

        raw = await upload.read()
        dest = session_dir / upload.filename
        dest.write_bytes(raw)

        audio, sr = librosa.load(str(dest), sr=None, mono=True)
        label, confidence = classify_stem(audio, sr, upload.filename)

        results.append({
            "filename": upload.filename,
            "label":    label,
            "confidence": round(confidence, 2),
            "duration_sec": round(len(audio) / sr, 2),
            "sample_rate":  sr,
        })

    return {"session_id": session_id, "stems": results}


@app.post("/mix/{session_id}")
async def mix_session(session_id: str):
    """
    Step 2: Mix the stems from a previously analyzed session.
    Returns a download URL for the final WAV.
    """
    session_dir = UPLOAD_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(404, "Session not found. Upload stems first via /analyze.")

    stem_files = list(session_dir.iterdir())
    if not stem_files:
        raise HTTPException(400, "No stems found in this session.")

    stems = []
    for path in stem_files:
        audio, sr = librosa.load(str(path), sr=44100, mono=False)
        if audio.ndim == 1:
            audio = np.stack([audio, audio])  # mono → stereo
        label, confidence = classify_stem(audio[0], sr, path.name)
        stems.append({"audio": audio, "sr": sr, "label": label, "filename": path.name})

    mixed, sr = mix_stems(stems)

    out_filename = f"mixlo_mix_{session_id[:8]}.wav"
    out_path = OUTPUT_DIR / out_filename
    sf.write(str(out_path), mixed.T, sr, subtype="PCM_24")

    return {"download_url": f"/download/{out_filename}", "filename": out_filename}


@app.get("/download/{filename}")
def download_mix(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found.")
    return FileResponse(str(path), media_type="audio/wav", filename=filename)

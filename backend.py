"""
backend.py — FastAPI backend for Better Shazam frontend demo.

POST /identify  — accepts WAV upload, streams SSE pipeline events + final result
GET  /health    — liveness check

Usage:
    python backend.py
    uvicorn backend:app --host 0.0.0.0 --port 8000
"""

import asyncio
import base64
import io
import json
import os
import queue
import re
import tempfile
import threading
import urllib.parse
import urllib.request
from math import gcd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from audio.preprocess import bandpass_filter, separate_stems, wiener_filter
from fingerprint.spectrogram import compute_mel_spectrogram, extract_peaks
from fingerprint.hash import generate_hashes
from fingerprint.database import query_db
from ml.embeddings import query_faiss
from config import FMAX, FMIN, HOP_LENGTH, SAMPLE_RATE

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DB_PATH    = os.path.join("output", "songs.db")
INDEX_PATH = os.path.join("ml", "faiss.index")
META_PATH  = os.path.join("ml", "metadata.json")
MODEL_PATH = os.path.join("ml", "embedder.pt")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Better Shazam")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sse(data: dict) -> str:
    """Format a dict as a Server-Sent Event frame."""
    return f"data: {json.dumps(data)}\n\n"


def _normalize(s: str) -> str:
    """Lowercase and strip all non-alphanumeric characters for title comparison."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _fetch_artwork(artist: str, title: str) -> str | None:
    """Fetch album art URL from the iTunes Search API. Returns None on failure."""
    try:
        term = urllib.parse.quote(f"{artist} {title}")
        url = f"https://itunes.apple.com/search?term={term}&entity=song&limit=1"
        with urllib.request.urlopen(url, timeout=5) as resp:
            payload = json.loads(resp.read())
        results = payload.get("results", [])
        if results:
            art = results[0].get("artworkUrl100", "")
            if art:
                return art.replace("100x100bb", "600x600bb")
    except Exception:
        pass
    return None


def _spectrogram_b64(audio: np.ndarray, sr: int) -> str:
    """Render a mel spectrogram to a base64-encoded PNG string."""
    spec = compute_mel_spectrogram(audio, sr=sr)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        spec,
        sr=sr,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        fmax=FMAX,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB", label="Amplitude (dB)")
    ax.set_title("Query Spectrogram", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _combine(
    classical_result,   # (title, artist, confidence) or None
    ml_result,          # list of (title, artist, score) or None
) -> tuple[str | None, str | None, str | None]:
    """Mirror matcher.py combination logic. Returns (title, artist, label)."""
    CLASSICAL_STRONG = 35
    ML_HIGH_SIM = 0.70

    c_title  = classical_result[0] if classical_result else None
    c_artist = classical_result[1] if classical_result else None
    c_conf   = classical_result[2] if classical_result else 0

    ml_top = ml_result[0] if ml_result else None
    ml_top_title  = ml_top[0] if ml_top else None
    ml_top_artist = ml_top[1] if ml_top else None

    # Strong classical signal — trust outright
    if c_title and c_conf >= CLASSICAL_STRONG:
        return c_title, c_artist, "CONFIDENT — strong classical match"

    # Check if classical title appears anywhere in the ML top-3
    ml_match_pos   = None
    ml_match_score = 0.0
    if c_title and ml_result:
        c_norm = _normalize(c_title)
        for pos, (ml_t, _, ml_s) in enumerate(ml_result):
            if _normalize(ml_t) == c_norm:
                ml_match_pos   = pos
                ml_match_score = ml_s
                break

    if ml_match_pos == 0 and ml_match_score >= ML_HIGH_SIM:
        return c_title, c_artist, "HIGH CONFIDENCE — both paths agree"
    elif ml_match_pos == 0:
        return c_title, c_artist, "MODERATE CONFIDENCE — paths agree, low ML similarity"
    elif ml_match_pos in (1, 2):
        return c_title, c_artist, "MODERATE CONFIDENCE — classical confirmed by ML"
    elif c_title and not ml_top_title:
        return c_title, c_artist, "MODERATE CONFIDENCE — classical only"
    elif ml_top_title and not c_title:
        return ml_top_title, ml_top_artist, "MODERATE CONFIDENCE — ML only"
    elif c_title and ml_top_title:
        return c_title, c_artist, "LOW CONFIDENCE — paths disagree (classical result returned)"
    else:
        return None, None, None


# ---------------------------------------------------------------------------
# Pipeline worker (runs in a background thread)
# ---------------------------------------------------------------------------

def _pipeline_worker(audio: np.ndarray, sr: int, eq: queue.Queue) -> None:
    """
    Execute the full identification pipeline, pushing SSE event dicts to eq.
    Sends None as a sentinel when finished (success or error).
    """
    try:
        # ── Demucs ──────────────────────────────────────────────────────────
        eq.put({"step": "demucs", "status": "started"})
        audio = separate_stems(audio, sr)
        eq.put({"step": "demucs", "status": "complete"})

        # ── Wiener ──────────────────────────────────────────────────────────
        eq.put({"step": "wiener", "status": "started"})
        audio = wiener_filter(audio)
        eq.put({"step": "wiener", "status": "complete"})

        # ── Bandpass ─────────────────────────────────────────────────────────
        eq.put({"step": "bandpass", "status": "started"})
        audio = bandpass_filter(audio, sr)
        eq.put({"step": "bandpass", "status": "complete"})

        cleaned = audio.copy()

        # ── Peak extraction (feeds classical path) ───────────────────────────
        eq.put({"step": "peaks", "status": "started"})
        spec   = compute_mel_spectrogram(cleaned, sr=sr)
        peaks  = extract_peaks(spec, sr=sr)
        hashes = generate_hashes(peaks)
        eq.put({"step": "peaks", "status": "complete"})

        # ── Spectrogram image (generated from cleaned audio, sent to frontend) ─
        spec_b64_val = _spectrogram_b64(cleaned, sr)
        eq.put({"step": "spectrogram", "status": "complete", "data": {"image": spec_b64_val}})

        # ── Classical + ML in parallel ───────────────────────────────────────
        classical_result: list = [None]
        ml_result: list        = [None]

        def run_classical():
            try:
                row = query_db(hashes, db_path=DB_PATH, min_confidence=1)
                if row:
                    _, title, artist, confidence = row
                    if confidence >= 5:
                        classical_result[0] = (title, artist, confidence)
            except Exception as e:
                print(f"  Classical error: {e}")

        def run_ml():
            if not (os.path.exists(INDEX_PATH) and os.path.exists(MODEL_PATH)):
                return
            try:
                results = query_faiss(
                    cleaned,
                    model_path=MODEL_PATH,
                    index_path=INDEX_PATH,
                    metadata_path=META_PATH,
                    top_k=3,
                )
                if results:
                    ml_result[0] = results
            except Exception as e:
                print(f"  ML error: {e}")

        # Emit both started events before threads begin
        eq.put({"step": "classical", "status": "started"})
        eq.put({"step": "ml",        "status": "started"})

        t_c = threading.Thread(target=run_classical)
        t_m = threading.Thread(target=run_ml)
        t_c.start()
        t_m.start()
        t_c.join()
        t_m.join()

        # Classical complete event
        c_data: dict = {}
        if classical_result[0]:
            c_title, _, c_conf = classical_result[0]
            c_data = {"title": c_title, "confidence": c_conf}
        eq.put({"step": "classical", "status": "complete", "data": c_data})

        # ML complete event
        ml_data: dict = {"results": []}
        if ml_result[0]:
            ml_data = {
                "results": [
                    {"title": t, "artist": a, "similarity": round(s, 3)}
                    for t, a, s in ml_result[0]
                ]
            }
        eq.put({"step": "ml", "status": "complete", "data": ml_data})

        # ── Combine ──────────────────────────────────────────────────────────
        title, artist, confidence_label = _combine(classical_result[0], ml_result[0])

        if title is None:
            eq.put({
                "step": "result",
                "status": "complete",
                "data": {
                    "title": None, "artist": None,
                    "confidence": "NO MATCH",
                    "classical": c_data, "ml": ml_data,
                    "artwork_url": None,
                },
            })
            return

        # ── Artwork fetch ─────────────────────────────────────────────────────
        artwork_url: list = [None]

        def fetch_art():
            artwork_url[0] = _fetch_artwork(artist, title)

        t_art = threading.Thread(target=fetch_art)
        t_art.start()
        t_art.join()

        eq.put({
            "step": "result",
            "status": "complete",
            "data": {
                "title":       title,
                "artist":      artist,
                "confidence":  confidence_label,
                "classical":   c_data,
                "ml":          ml_data,
                "artwork_url": artwork_url[0],
            },
        })

    except Exception as e:
        eq.put({"step": "error", "status": "error", "data": {"message": str(e)}})
    finally:
        eq.put(None)  # sentinel — tells the SSE stream to close


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    """Accept a WAV file upload and stream SSE events for each pipeline step."""
    contents = await file.read()

    # Write to a temp file so soundfile can read the header
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        audio, sr = sf.read(tmp_path, dtype="float32")
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Ensure mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample to SAMPLE_RATE if needed
    if sr != SAMPLE_RATE:
        g     = gcd(SAMPLE_RATE, sr)
        audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)

    eq: queue.Queue = queue.Queue()

    threading.Thread(
        target=_pipeline_worker,
        args=(audio, SAMPLE_RATE, eq),
        daemon=True,
    ).start()

    async def stream():
        loop = asyncio.get_event_loop()
        while True:
            # Block in a thread-pool executor so the event loop stays free
            event = await loop.run_in_executor(None, eq.get)
            if event is None:
                break
            yield _sse(event)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False)

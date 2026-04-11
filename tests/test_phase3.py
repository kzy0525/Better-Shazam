"""
tests/test_phase3.py — Phase 3 real-song demo.

Scans a folder of audio files (MP3, AAC, FLAC, WAV, M4A, etc.),
converts each to 22050 Hz mono WAV using ffmpeg, registers them all
in a fresh SQLite database, then records a 10-second mic snippet and
identifies which song is playing.

Usage:
    # Point at a folder of audio files
    python tests/test_phase3.py --songs /path/to/music

    # Optionally reuse an existing DB (skip re-registration)
    python tests/test_phase3.py --songs /path/to/music --reuse-db

    # Set minimum confidence threshold (default 5)
    python tests/test_phase3.py --songs /path/to/music --min-confidence 10
"""

import sys
import os
import argparse
import subprocess
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import soundfile as sf

from audio.capture import record
from audio.preprocess import preprocess
from fingerprint.spectrogram import compute_mel_spectrogram, extract_peaks
from fingerprint.hash import generate_hashes
from fingerprint.database import register_song, query_db, list_songs, create_db
from config import SAMPLE_RATE, CLIP_LENGTH

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
DB_PATH = os.path.join(OUT_DIR, "songs.db")
WAV_CACHE_DIR = os.path.join(OUT_DIR, "wav_cache")

SUPPORTED_EXTENSIONS = {".mp3", ".m4a", ".aac", ".flac", ".ogg", ".wav", ".aiff"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_audio_files(folder: str):
    """Recursively find all supported audio files under folder."""
    found = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS:
                found.append(os.path.join(root, f))
    return sorted(found)


def convert_to_wav(src: str, dst: str) -> None:
    """Convert any audio file to 22050 Hz mono WAV using ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y",           # overwrite without asking
            "-i", src,
            "-ar", str(SAMPLE_RATE), # resample to 22050 Hz
            "-ac", "1",              # mono
            "-sample_fmt", "s16",    # 16-bit PCM
            dst,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def slugify(name: str) -> str:
    """Turn a filename into a safe slug for the WAV cache."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def extract_metadata(filepath: str):
    """Best-effort title/artist extraction from filename."""
    stem = os.path.splitext(os.path.basename(filepath))[0]
    # Common pattern: "Artist - Title"
    if " - " in stem:
        parts = stem.split(" - ", 1)
        return parts[1].strip(), parts[0].strip()
    return stem, "Unknown"


def pipeline(audio, sr: int = SAMPLE_RATE):
    """Run a numpy audio array through the full fingerprinting pipeline."""
    audio = preprocess(audio, sample_rate=sr)
    spec = compute_mel_spectrogram(audio, sr=sr)
    peaks = extract_peaks(spec, sr=sr)
    return generate_hashes(peaks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 real-song demo")
    parser.add_argument("--songs", required=True, help="Folder containing audio files")
    parser.add_argument("--reuse-db", action="store_true", help="Skip re-registering songs if DB exists")
    parser.add_argument("--min-confidence", type=int, default=15, help="Minimum match confidence (default: 15)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(WAV_CACHE_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Find audio files
    # ------------------------------------------------------------------
    audio_files = find_audio_files(args.songs)
    if not audio_files:
        print(f"No supported audio files found in: {args.songs}")
        sys.exit(1)
    print(f"Found {len(audio_files)} audio file(s) in {args.songs}")

    # ------------------------------------------------------------------
    # 2. Convert + register songs
    # ------------------------------------------------------------------
    if args.reuse_db and os.path.exists(DB_PATH):
        print(f"Reusing existing database: {DB_PATH}")
    else:
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        create_db(DB_PATH)

        print(f"\n=== Converting and registering {len(audio_files)} song(s) ===")
        for src in audio_files:
            title, artist = extract_metadata(src)
            slug = slugify(os.path.splitext(os.path.basename(src))[0])
            wav_path = os.path.join(WAV_CACHE_DIR, f"{slug}.wav")

            ext = os.path.splitext(src)[1].lower()
            if ext == ".wav":
                # Check if already the right format
                info = sf.info(src)
                if info.samplerate == SAMPLE_RATE and info.channels == 1:
                    wav_path = src  # use as-is
                else:
                    print(f"  Converting (resample/mono): {os.path.basename(src)}")
                    convert_to_wav(src, wav_path)
            else:
                print(f"  Converting: {os.path.basename(src)}")
                convert_to_wav(src, wav_path)

            register_song(title, artist, wav_path, db_path=DB_PATH)

    # ------------------------------------------------------------------
    # 3. List registered songs
    # ------------------------------------------------------------------
    print("\n=== Registered songs ===")
    list_songs(db_path=DB_PATH)

    # ------------------------------------------------------------------
    # 4. Record a snippet and identify it
    # ------------------------------------------------------------------
    print(f"=== Recording {int(CLIP_LENGTH)}s snippet — play a song now! ===")
    snippet_path = os.path.join(OUT_DIR, "query_snippet.wav")
    snippet = record(duration=CLIP_LENGTH, sample_rate=SAMPLE_RATE, save_path=snippet_path)

    print("\nFingerprinting snippet...")
    query_hashes = pipeline(snippet)
    print(f"Generated {len(query_hashes)} query hashes")

    # ------------------------------------------------------------------
    # 5. Query and print result
    # ------------------------------------------------------------------
    print("\n=== Match result ===")
    result = query_db(query_hashes, db_path=DB_PATH, min_confidence=args.min_confidence)

    if result is None:
        print("No match found. Try:")
        print("  - Playing the audio louder / closer to the mic")
        print(f"  - Lowering --min-confidence (currently {args.min_confidence})")
        print("  - Making sure the song is registered in the database")
    else:
        song_id, title, artist, confidence = result
        print(f"  Matched : {title}")
        print(f"  Artist  : {artist}")
        print(f"  Song ID : {song_id}")
        print(f"  Confidence: {confidence} aligned hash pairs")


if __name__ == "__main__":
    main()

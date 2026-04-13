"""
tests/test_phase5.py — Phase 5 end-to-end verification.

1. Converts test_songs/ MP3s to WAV (if not already done)
2. Registers songs in the classical DB (Phase 3)
3. Trains AudioEmbedder with triplet loss
4. Builds FAISS index
5. Evaluates tempo robustness — key portfolio artifact
6. Records a 10s snippet → runs through matcher.match (both paths)
7. Tests a tempo-shifted version to confirm ML path handles it
8. Visualises embeddings via PCA
9. Prints final pass/fail summary

Usage:
    python tests/test_phase5.py
    python tests/test_phase5.py --skip-train   # reuse existing model
    python tests/test_phase5.py --skip-record  # use saved snippet
"""

import sys
import os
import argparse
import subprocess
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import soundfile as sf
import audiomentations as A
import torch
import torch.nn.functional as F

from fingerprint.database import register_song, create_db, list_songs
from ml.dataset import build_triplet_dataset, verify_dataset, slice_audio, _to_spectrogram, _spec_to_tensor
from ml.model import train, get_model, evaluate_tempo_robustness
from ml.embeddings import build_faiss_index, query_faiss, visualize_embeddings
from audio.capture import record
from audio.preprocess import full_preprocess_pipeline
from matcher import match
from config import SAMPLE_RATE, CLIP_LENGTH

# Paths
ROOT       = os.path.join(os.path.dirname(__file__), "..")
SONGS_DIR  = os.path.join(ROOT, "songs")
WAV_DIR    = os.path.join(ROOT, "output", "wav_cache")
OUT_DIR    = os.path.join(ROOT, "output")
ML_DIR     = os.path.join(ROOT, "ml")
DB_PATH    = os.path.join(OUT_DIR, "songs.db")
MODEL_PATH = os.path.join(ML_DIR, "embedder.pt")
INDEX_PATH = os.path.join(ML_DIR, "faiss.index")
META_PATH  = os.path.join(ML_DIR, "metadata.json")
SNIPPET_PATH = os.path.join(OUT_DIR, "test_phase5_snippet.wav")

SUPPORTED = {".mp3", ".m4a", ".aac", ".flac", ".ogg", ".wav"}


def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def extract_metadata(filepath: str):
    stem = os.path.splitext(os.path.basename(filepath))[0]
    # Handle both "Artist - Title" (original) and "Artist_-_Title" (slugified)
    for sep in [" - ", "_-_"]:
        if sep in stem:
            parts = stem.split(sep, 1)
            artist = parts[0].replace("_", " ").strip()
            title = parts[1].replace("_", " ").strip()
            return title, artist
    return stem.replace("_", " "), "Unknown"


def convert_to_wav(src: str, dst: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ar", str(SAMPLE_RATE), "-ac", "1", "-sample_fmt", "s16", dst],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def find_audio_files(folder: str):
    return sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED
    ])


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-register", action="store_true", help="Skip step 1 — reuse existing songs.db and wav_cache")
    parser.add_argument("--skip-train",    action="store_true", help="Skip step 3 — reuse existing model weights")
    parser.add_argument("--skip-record",   action="store_true", help="Skip step 6 — reuse saved snippet")
    parser.add_argument("--epochs",        type=int, default=5)
    args = parser.parse_args()

    os.makedirs(WAV_DIR, exist_ok=True)
    os.makedirs(ML_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    results = {}

    # ------------------------------------------------------------------
    # 1. Convert + register songs (classical DB)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1 — Converting and registering songs" + (" (SKIPPED)" if args.skip_register else ""))
    print("=" * 60)

    # Always collect wav_paths from wav_cache so later steps can use them
    wav_paths = sorted([
        os.path.join(WAV_DIR, f)
        for f in os.listdir(WAV_DIR)
        if f.lower().endswith(".wav")
    ]) if args.skip_register and os.path.isdir(WAV_DIR) else []

    if not args.skip_register:
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        create_db(DB_PATH)

        audio_files = find_audio_files(SONGS_DIR)
        for src in audio_files:
            title, artist = extract_metadata(src)
            slug = slugify(os.path.splitext(os.path.basename(src))[0])
            wav_path = os.path.join(WAV_DIR, f"{slug}.wav")
            ext = os.path.splitext(src)[1].lower()
            if ext != ".wav":
                print(f"  Converting: {os.path.basename(src)}")
                convert_to_wav(src, wav_path)
            else:
                info = sf.info(src)
                if info.samplerate == SAMPLE_RATE and info.channels == 1:
                    wav_path = src
                else:
                    convert_to_wav(src, wav_path)
            register_song(title, artist, wav_path, db_path=DB_PATH)
            wav_paths.append(wav_path)

    print()
    list_songs(db_path=DB_PATH)

    # ------------------------------------------------------------------
    # 2. Build triplet dataset + verify
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 2 — Building triplet dataset")
    print("=" * 60)
    dataset = build_triplet_dataset(WAV_DIR, clips_per_song=3, augmentations_per_clip=3)
    verify_dataset(dataset, n_samples=3, save_path=os.path.join(ML_DIR, "dataset_verify.png"))

    # ------------------------------------------------------------------
    # 3. Train (or load) model
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 3 — Training AudioEmbedder" + (" (SKIPPED)" if args.skip_train else ""))
    print("=" * 60)
    if args.skip_train and os.path.exists(MODEL_PATH):
        model = get_model(MODEL_PATH)
        model.eval()
    else:
        model = train(
            gtzan_dir=os.path.join(ROOT, "data", "genres_original"),
            epochs=args.epochs,
            batch_size=64,
            lr=1e-4,
            save_path=MODEL_PATH,
            genres=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop"],
        )

    # ------------------------------------------------------------------
    # 4. Tempo robustness evaluation
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 4 — Tempo robustness evaluation")
    print("=" * 60)
    evaluate_tempo_robustness(
        model,
        songs_dir=WAV_DIR,
        save_path=os.path.join(ML_DIR, "tempo_robustness.png"),
    )
    results["tempo_robustness"] = "✓ completed"

    # ------------------------------------------------------------------
    # 5. Build FAISS index
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 5 — Building FAISS index")
    print("=" * 60)
    build_faiss_index(
        songs_dir=WAV_DIR,
        model_path=MODEL_PATH,
        index_path=INDEX_PATH,
        metadata_path=META_PATH,
    )
    results["faiss_index"] = "✓ built"

    # ------------------------------------------------------------------
    # 6. Record snippet + run dual-path matcher
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 6 — Record snippet and run matcher")
    print("=" * 60)

    if args.skip_record and os.path.exists(SNIPPET_PATH):
        print(f"Reusing saved snippet: {SNIPPET_PATH}")
        snippet, _ = sf.read(SNIPPET_PATH, dtype="float32")
    else:
        print(f"Recording {int(CLIP_LENGTH)}s — play one of your registered songs now!")
        snippet = record(duration=CLIP_LENGTH, sample_rate=SAMPLE_RATE, save_path=SNIPPET_PATH)

    result = match(
        snippet,
        sr=SAMPLE_RATE,
        db_path=DB_PATH,
        index_path=INDEX_PATH,
        metadata_path=META_PATH,
        model_path=MODEL_PATH,
    )
    results["live_match"] = f"✓ {result[0]} — {result[2]}" if result else "✗ no match"

    # ------------------------------------------------------------------
    # 7. Tempo-shifted version — does ML path still match?
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 7 — Tempo-shifted snippet (1.20x) via ML path")
    print("=" * 60)

    # Use mid-clip from the first registered song
    ref_audio, ref_sr = sf.read(wav_paths[0], dtype="float32")
    if ref_audio.ndim > 1:
        ref_audio = ref_audio.mean(axis=1)
    mid = len(ref_audio) // 2
    half = int(CLIP_LENGTH * ref_sr // 2)
    clean_clip = ref_audio[mid - half: mid + half].astype(np.float32)

    stretcher = A.TimeStretch(min_rate=1.20, max_rate=1.20, p=1.0)
    shifted = stretcher(samples=clean_clip, sample_rate=ref_sr)
    shifted = shifted[:len(clean_clip)] if len(shifted) > len(clean_clip) \
        else np.pad(shifted, (0, len(clean_clip) - len(shifted)))
    shifted = shifted.astype(np.float32)

    print("Running ML path on 1.20x speed clip...")
    ml_results = query_faiss(shifted, model_path=MODEL_PATH, index_path=INDEX_PATH,
                             metadata_path=META_PATH, top_k=3)
    expected_title, expected_artist = extract_metadata(wav_paths[0])
    if ml_results:
        top_title, top_artist, top_score = ml_results[0]
        passed = top_title.lower() == expected_title.lower()
        status = "✓" if passed else "✗"
        print(f"  Expected : {expected_title}")
        print(f"  Got      : {top_title} (similarity={top_score:.3f}) {status}")
        results["tempo_shift_ml"] = f"{status} {top_title} sim={top_score:.3f}"
    else:
        print("  ML path returned no results for shifted clip")
        results["tempo_shift_ml"] = "✗ no match"

    # ------------------------------------------------------------------
    # 8. PCA embedding visualisation
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 8 — PCA embedding visualisation")
    print("=" * 60)
    visualize_embeddings(
        songs_dir=WAV_DIR,
        model_path=MODEL_PATH,
        save_path=os.path.join(ML_DIR, "embeddings_pca.png"),
    )
    results["pca_plot"] = "✓ saved"

    # ------------------------------------------------------------------
    # 9. Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 5 SUMMARY")
    print("=" * 60)
    for test, outcome in results.items():
        print(f"  {test:<25} {outcome}")
    print("=" * 60)


if __name__ == "__main__":
    main()

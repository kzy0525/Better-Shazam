"""
matcher.py — Dual-path song identification.

Runs classical fingerprinting (Phase 3) and ML embedding search (Phase 5)
in parallel on the same preprocessed audio, then combines their results
into a single confidence-weighted decision.
"""

import os
import re
import threading
from typing import Optional, Tuple

import numpy as np

from audio.preprocess import full_preprocess_pipeline
from fingerprint.spectrogram import compute_mel_spectrogram, extract_peaks
from fingerprint.hash import generate_hashes
from fingerprint.database import query_db
from ml.embeddings import query_faiss
from config import SAMPLE_RATE


def match(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    db_path: str = "output/songs.db",
    index_path: str = "ml/faiss.index",
    metadata_path: str = "ml/metadata.json",
    model_path: str = "ml/embedder.pt",
) -> Optional[Tuple[str, str, str]]:
    """Run both fingerprinting paths in parallel and combine results.

    Classical path: audio → bandpass → spectrogram → peaks → hashes → SQLite
    ML path:        audio → Demucs → Wiener → spectrogram → embedding → FAISS

    Combination logic:
        Both agree      → HIGH CONFIDENCE — both paths agree
        Classical only  → MODERATE CONFIDENCE — classical only
        ML only         → MODERATE CONFIDENCE — ML only
        Neither matches → NO MATCH FOUND

    Args:
        audio:         1-D float32 numpy array (raw, not yet preprocessed).
        sr:            Sample rate in Hz.
        db_path:       Path to the SQLite fingerprint database.
        index_path:    Path to the FAISS index.
        metadata_path: Path to the FAISS metadata JSON.
        model_path:    Path to the trained AudioEmbedder weights.

    Returns:
        (title, artist, confidence_label) tuple, or None if no match found.
    """
    # Preprocess once — both paths share the same cleaned audio
    print("Preprocessing audio...")
    cleaned = full_preprocess_pipeline(audio, sample_rate=sr)

    classical_result: list = [None]
    ml_result: list = [None]

    def run_classical():
        try:
            spec = compute_mel_spectrogram(cleaned, sr=sr)
            peaks = extract_peaks(spec, sr=sr)
            hashes = generate_hashes(peaks)
            result = query_db(hashes, db_path=db_path, min_confidence=1)
            if result:
                _, title, artist, confidence = result
                print(f"  Classical raw: '{title}' confidence={confidence}")
                if confidence >= 5:
                    classical_result[0] = (title, artist, confidence)
        except Exception as e:
            print(f"  Classical path error: {e}")

    def run_ml():
        if not (os.path.exists(index_path) and os.path.exists(model_path)):
            return
        try:
            results = query_faiss(
                cleaned,
                model_path=model_path,
                index_path=index_path,
                metadata_path=metadata_path,
                top_k=3,
            )
            if results:
                ml_result[0] = results  # list of (title, artist, score)
        except Exception as e:
            print(f"  ML path error: {e}")

    # Run both paths concurrently
    t_classical = threading.Thread(target=run_classical)
    t_ml = threading.Thread(target=run_ml)
    t_classical.start()
    t_ml.start()
    t_classical.join()
    t_ml.join()

    def _normalize(s: str) -> str:
        """Lowercase, strip punctuation, collapse whitespace for title comparison."""
        s = s.lower()
        s = re.sub(r"[^\w\s]", "", s)
        return re.sub(r"\s+", " ", s).strip()

    # -----------------------------------------------------------------------
    # Report individual path results
    # -----------------------------------------------------------------------
    print("\n--- Path Results ---")

    c_title, c_artist, c_conf = None, None, 0
    if classical_result[0]:
        c_title, c_artist, c_conf = classical_result[0]
        print(f"  Classical : '{c_title}' by {c_artist}  (confidence={c_conf})")
    else:
        print("  Classical : no match")

    if ml_result[0]:
        ml_top_title, ml_top_artist, ml_top_score = ml_result[0][0]
        print(f"  ML        : '{ml_top_title}' by {ml_top_artist}  (similarity={ml_top_score:.3f})")
        for title, artist, score in ml_result[0][1:]:
            print(f"              '{title}' by {artist}  (similarity={score:.3f})")
    else:
        ml_top_title, ml_top_artist, ml_top_score = None, None, 0.0

    # -----------------------------------------------------------------------
    # Combine — check if classical matches any of the top 3 ML results
    # -----------------------------------------------------------------------
    ML_HIGH_SIMILARITY = 0.70

    print("\n--- Final Decision ---")

    ml_match_position = None  # 0-indexed position in ML top-3 where classical agrees
    ml_match_score = 0.0
    if c_title is not None and ml_result[0]:
        c_norm = _normalize(c_title)
        for pos, (ml_t, _, ml_s) in enumerate(ml_result[0]):
            if _normalize(ml_t) == c_norm:
                ml_match_position = pos
                ml_match_score = ml_s
                break

    if ml_match_position == 0 and ml_match_score >= ML_HIGH_SIMILARITY:
        label = "HIGH CONFIDENCE — both paths agree"
        title, artist = c_title, c_artist
    elif ml_match_position == 0:
        label = "MODERATE CONFIDENCE — paths agree, low ML similarity"
        title, artist = c_title, c_artist
    elif ml_match_position in (1, 2):
        label = "MODERATE CONFIDENCE — classical confirmed by ML"
        title, artist = c_title, c_artist
    elif c_title is not None and ml_top_title is None:
        label = "MODERATE CONFIDENCE — classical only"
        title, artist = c_title, c_artist
    elif ml_top_title is not None and c_title is None:
        label = "MODERATE CONFIDENCE — ML only"
        title, artist = ml_top_title, ml_top_artist
    elif c_title is not None and ml_top_title is not None:
        # Both returned results but none of the top 3 ML results matched classical
        label = "LOW CONFIDENCE — paths disagree (classical result returned)"
        title, artist = c_title, c_artist
    else:
        print("  NO MATCH FOUND")
        return None

    print(f"  {title} by {artist}")
    print(f"  {label}")
    return title, artist, label

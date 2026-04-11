"""
fingerprint/database.py — SQLite fingerprint store.

Handles song registration (full pipeline: WAV → hashes → DB) and
query matching via time-coherence histogramming.
"""

import sqlite3
from collections import defaultdict
from typing import List, Tuple

import soundfile as sf

from audio.preprocess import full_preprocess_pipeline
from fingerprint.spectrogram import compute_mel_spectrogram, extract_peaks
from fingerprint.hash import generate_hashes
from config import SAMPLE_RATE


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS songs (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    title    TEXT NOT NULL,
    artist   TEXT NOT NULL,
    filepath TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fingerprints (
    hash        INTEGER NOT NULL,
    time_offset INTEGER NOT NULL,
    song_id     INTEGER NOT NULL,
    FOREIGN KEY (song_id) REFERENCES songs(id)
);

CREATE INDEX IF NOT EXISTS idx_hash ON fingerprints(hash);
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_db(db_path: str = "shazam.db") -> None:
    """Create the SQLite database and tables if they do not already exist.

    Args:
        db_path: Path to the SQLite database file.
    """
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_DDL)
    print(f"Database ready: {db_path}")


def register_song(
    title: str,
    artist: str,
    filepath: str,
    db_path: str = "shazam.db",
) -> int:
    """Register a WAV file in the database by running it through the full pipeline.

    Pipeline:
        WAV → noise reduction → bandpass filter → mel spectrogram → peak extraction → hashing → SQLite

    Args:
        title:    Song title string.
        artist:   Artist name string.
        filepath: Path to a WAV file (will be read at sr=22050 mono).
        db_path:  Path to the SQLite database file.

    Returns:
        The auto-assigned song_id integer.
    """
    # Load audio
    audio, sr = sf.read(filepath, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr} Hz for {filepath}")

    # Full fingerprinting pipeline
    audio = full_preprocess_pipeline(audio, sample_rate=sr)
    spec = compute_mel_spectrogram(audio, sr=sr)
    peaks = extract_peaks(spec, sr=sr)
    hashes = generate_hashes(peaks)

    # Write to DB
    create_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO songs (title, artist, filepath) VALUES (?, ?, ?)",
            (title, artist, filepath),
        )
        song_id = cursor.lastrowid
        conn.executemany(
            "INSERT INTO fingerprints (hash, time_offset, song_id) VALUES (?, ?, ?)",
            [(h, t, song_id) for h, t in hashes],
        )

    print(f"Registered '{title}' by {artist} — song_id={song_id}, hashes stored={len(hashes)}")
    return song_id


def query_db(
    query_hashes: List[Tuple[int, int]],
    db_path: str = "shazam.db",
    min_confidence: int = 5,
):
    """Match a snippet's hashes against the database using time-coherence voting.

    For each matching hash, records (db_time_offset - query_time_offset) per
    candidate song. A genuine match will produce many pairs with the same delta
    (the playback position in the song). The song with the tallest histogram bin
    wins, provided it exceeds min_confidence.

    Args:
        query_hashes:   List of (hash_int, time_offset) tuples from the snippet.
        db_path:        Path to the SQLite database file.
        min_confidence: Minimum bin count required to declare a match.

    Returns:
        (song_id, title, artist, confidence) tuple, or None if no match found.
    """
    if not query_hashes:
        return None

    # Build a lookup: hash_int → list of query time offsets
    query_lookup: dict[int, List[int]] = defaultdict(list)
    for h, t in query_hashes:
        query_lookup[h].append(t)

    # Fetch all matching DB rows in one query
    hash_ints = list(query_lookup.keys())
    placeholders = ",".join("?" * len(hash_ints))

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT hash, time_offset, song_id FROM fingerprints WHERE hash IN ({placeholders})",
            hash_ints,
        ).fetchall()

        # Build time-delta histograms: song_id → {delta → count}
        histograms: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for db_hash, db_offset, song_id in rows:
            for query_offset in query_lookup[db_hash]:
                delta = db_offset - query_offset
                histograms[song_id][delta] += 1

        if not histograms:
            return None

        # Find the (song_id, delta) pair with the highest vote count
        best_song_id = max(
            histograms,
            key=lambda sid: max(histograms[sid].values()),
        )
        confidence = max(histograms[best_song_id].values())

        if confidence < min_confidence:
            return None

        row = conn.execute(
            "SELECT title, artist FROM songs WHERE id = ?", (best_song_id,)
        ).fetchone()

    title, artist = row
    return (best_song_id, title, artist, confidence)


def list_songs(db_path: str = "shazam.db") -> None:
    """Print all songs registered in the database.

    Args:
        db_path: Path to the SQLite database file.
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT id, title, artist, filepath FROM songs").fetchall()

    if not rows:
        print("No songs registered.")
        return

    print(f"\n{'ID':<5} {'Title':<30} {'Artist':<25} Filepath")
    print("-" * 80)
    for song_id, title, artist, filepath in rows:
        print(f"{song_id:<5} {title:<30} {artist:<25} {filepath}")
    print()

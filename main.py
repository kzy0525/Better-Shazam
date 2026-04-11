"""
main.py — Entry point for the Shazam clone.

Records 10 seconds from the microphone, runs the full dual-path
identification pipeline, and prints the result.

Usage:
    python main.py
    python main.py --db output/songs.db --index ml/faiss.index
"""

import argparse
import os

from audio.capture import record
from matcher import match
from config import SAMPLE_RATE, CLIP_LENGTH

DEFAULT_DB    = os.path.join("output", "songs.db")
DEFAULT_INDEX = os.path.join("ml", "faiss.index")
DEFAULT_META  = os.path.join("ml", "metadata.json")
DEFAULT_MODEL = os.path.join("ml", "embedder.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Shazam clone — identify a song from mic input")
    parser.add_argument("--db",       default=DEFAULT_DB,    help="Path to SQLite fingerprint DB")
    parser.add_argument("--index",    default=DEFAULT_INDEX, help="Path to FAISS index")
    parser.add_argument("--metadata", default=DEFAULT_META,  help="Path to FAISS metadata JSON")
    parser.add_argument("--model",    default=DEFAULT_MODEL, help="Path to AudioEmbedder weights")
    parser.add_argument("--duration", type=float, default=CLIP_LENGTH, help="Recording duration (s)")
    args = parser.parse_args()

    print("Calibrating...", flush=True)

    # Record
    print(f"\nListening... (recording {int(args.duration)}s — play a song now!)")
    audio = record(duration=args.duration, sample_rate=SAMPLE_RATE)

    # Identify
    result = match(
        audio,
        sr=SAMPLE_RATE,
        db_path=args.db,
        index_path=args.index,
        metadata_path=args.metadata,
        model_path=args.model,
    )

    # Print final result
    print()
    if result is None:
        print("Result : NO MATCH FOUND")
        print("  Try playing the audio louder or re-registering the song.")
    else:
        title, artist, confidence = result
        print("=" * 50)
        print(f"  Song   : {title}")
        print(f"  Artist : {artist}")
        print(f"  Status : {confidence}")
        print("=" * 50)


if __name__ == "__main__":
    main()

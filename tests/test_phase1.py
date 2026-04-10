"""
tests/test_phase1.py — Phase 1 visual verification.

Records 10 seconds from the microphone, runs it through the bandpass
filter, saves both the raw and filtered WAV files, then plots both
waveforms side by side so you can visually confirm the filter is working.

Usage:
    python tests/test_phase1.py
"""

import sys
import os

# Allow imports from the project root regardless of where the script is run
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from audio.capture import record
from audio.preprocess import bandpass_filter
from config import SAMPLE_RATE, CLIP_LENGTH, FMIN, FMAX

# Output paths
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
RAW_PATH = os.path.join(OUT_DIR, "test_raw.wav")
FILTERED_PATH = os.path.join(OUT_DIR, "test_filtered.wav")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- 1. Record -----------------------------------------------------------
    raw = record(duration=CLIP_LENGTH, sample_rate=SAMPLE_RATE, save_path=RAW_PATH)

    # --- 2. Filter -----------------------------------------------------------
    filtered = bandpass_filter(raw, sample_rate=SAMPLE_RATE, low_hz=FMIN, high_hz=FMAX)
    sf.write(FILTERED_PATH, filtered, SAMPLE_RATE)
    print(f"Saved filtered recording to: {FILTERED_PATH}")

    # --- 3. Plot -------------------------------------------------------------
    time_axis = np.linspace(0, len(raw) / SAMPLE_RATE, num=len(raw))

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle("Phase 1 — Waveform Comparison", fontsize=14, fontweight="bold")

    axes[0].plot(time_axis, raw, color="steelblue", linewidth=0.5)
    axes[0].set_title("Raw (unfiltered)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_ylim(-1.0, 1.0)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_axis, filtered, color="darkorange", linewidth=0.5)
    axes[1].set_title(f"Filtered (Butterworth bandpass {FMIN}–{FMAX} Hz)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylim(-1.0, 1.0)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "test_phase1_waveforms.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()

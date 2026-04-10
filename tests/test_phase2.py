"""
tests/test_phase2.py — Phase 2 visual verification.

Loads the filtered WAV saved by test_phase1.py, runs it through the
mel spectrogram pipeline, extracts peaks, and plots the spectrogram
twice — once clean, once with peaks overlaid.

Usage:
    python tests/test_phase2.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import librosa

from fingerprint.spectrogram import compute_mel_spectrogram, extract_peaks, plot_spectrogram
from config import SAMPLE_RATE

FILTERED_WAV = os.path.join(os.path.dirname(__file__), "..", "output", "test_filtered.wav")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def main() -> None:
    # --- 1. Load filtered WAV from Phase 1 -----------------------------------
    print(f"Loading: {FILTERED_WAV}")
    audio, sr = librosa.load(FILTERED_WAV, sr=SAMPLE_RATE, mono=True)
    print(f"Audio loaded — samples: {len(audio)}, sr: {sr} Hz, duration: {len(audio)/sr:.2f}s")

    # --- 2. Compute mel spectrogram ------------------------------------------
    spec = compute_mel_spectrogram(audio, sr=sr)
    print(f"Spectrogram shape: {spec.shape}  (mel_bins × time_frames)")

    # --- 3. Extract peaks ----------------------------------------------------
    peaks = extract_peaks(spec, sr=sr)
    print(f"Peaks found: {len(peaks)}")

    print("\nTop 5 strongest peaks (time_frame, freq_bin, amplitude_dB):")
    for i, (t, f, a) in enumerate(peaks[:5]):
        import librosa as _librosa
        freqs = _librosa.mel_frequencies(n_mels=128, fmin=80, fmax=8000)
        time_sec = _librosa.frames_to_time(t, sr=sr, hop_length=512)
        print(f"  #{i+1}  t={time_sec:.3f}s  freq={freqs[f]:.1f}Hz  amp={a:.1f}dB")

    # --- 4. Plot without peaks -----------------------------------------------
    plot_spectrogram(
        spec,
        sr=sr,
        peaks=None,
        title="Phase 2 — Mel Spectrogram (no peaks)",
        save_path=os.path.join(OUT_DIR, "test_phase2_spectrogram.png"),
    )

    # --- 5. Plot with peaks overlaid -----------------------------------------
    plot_spectrogram(
        spec,
        sr=sr,
        peaks=peaks,
        title="Phase 2 — Mel Spectrogram with Peaks",
        save_path=os.path.join(OUT_DIR, "test_phase2_peaks.png"),
    )


if __name__ == "__main__":
    main()

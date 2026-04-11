"""
tests/test_phase4.py — Phase 4 visual verification.

Records 10 seconds of audio with background noise present, runs it
through each stage of the preprocessing pipeline, and plots 3 waveforms
stacked vertically:
    1. Raw (unprocessed)
    2. After noise reduction only
    3. After full pipeline (noise reduction + bandpass)

Also computes and prints the RMS energy of the noise sample before and
after reduction, and plots the mel spectrogram of the fully cleaned audio.

Usage:
    python tests/test_phase4.py

Make sure there is some background noise present during recording
(fan, TV, ambient room noise) to see the effect of noise reduction.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from audio.capture import record
from audio.preprocess import reduce_noise, bandpass_filter, full_preprocess_pipeline
from fingerprint.spectrogram import compute_mel_spectrogram, plot_spectrogram
from config import SAMPLE_RATE, CLIP_LENGTH

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Record
    # ------------------------------------------------------------------
    print("Recording — make sure there is background noise present...")
    raw = record(duration=CLIP_LENGTH, sample_rate=SAMPLE_RATE)
    sf.write(os.path.join(OUT_DIR, "phase4_raw.wav"), raw, SAMPLE_RATE)

    # ------------------------------------------------------------------
    # 2. Stage-by-stage processing
    # ------------------------------------------------------------------
    noise_sample_duration = 0.5
    noise_samples = int(noise_sample_duration * SAMPLE_RATE)

    denoised_only = reduce_noise(raw, sample_rate=SAMPLE_RATE,
                                 noise_sample_duration=noise_sample_duration)
    fully_cleaned = bandpass_filter(denoised_only, sample_rate=SAMPLE_RATE)
    sf.write(os.path.join(OUT_DIR, "phase4_cleaned.wav"), fully_cleaned, SAMPLE_RATE)

    # ------------------------------------------------------------------
    # 3. RMS energy comparison
    # ------------------------------------------------------------------
    noise_before = raw[:noise_samples]
    noise_after  = denoised_only[:noise_samples]

    rms_before = rms(noise_before)
    rms_after  = rms(noise_after)
    reduction_db = 20 * np.log10(rms_after / rms_before + 1e-10)

    print(f"\nNoise sample RMS (first {noise_sample_duration}s):")
    print(f"  Before reduction : {rms_before:.6f}")
    print(f"  After reduction  : {rms_after:.6f}")
    print(f"  Reduction        : {reduction_db:.1f} dB")

    # ------------------------------------------------------------------
    # 4. Waveform plot — 3 subplots
    # ------------------------------------------------------------------
    time_axis = np.linspace(0, CLIP_LENGTH, num=len(raw))
    y_lim = max(np.abs(raw).max(), 0.1) * 1.1

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    fig.suptitle("Phase 4 — Preprocessing Pipeline Comparison", fontsize=14, fontweight="bold")

    stages = [
        (raw,           "steelblue",  "1. Raw (unprocessed)"),
        (denoised_only, "mediumorchid", "2. After noise reduction only"),
        (fully_cleaned, "darkorange",  "3. After full pipeline (noise reduction + bandpass)"),
    ]

    for ax, (audio, color, title) in zip(axes, stages):
        ax.plot(time_axis, audio, color=color, linewidth=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Amplitude")
        ax.set_ylim(-y_lim, y_lim)
        ax.axvspan(0, noise_sample_duration, alpha=0.12, color="red",
                   label=f"Noise profile ({noise_sample_duration}s)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    waveform_path = os.path.join(OUT_DIR, "test_phase4_waveforms.png")
    plt.savefig(waveform_path, dpi=150)
    print(f"\nWaveform plot saved to: {waveform_path}")
    plt.show()

    # ------------------------------------------------------------------
    # 5. Mel spectrogram of fully cleaned audio
    # ------------------------------------------------------------------
    spec = compute_mel_spectrogram(fully_cleaned, sr=SAMPLE_RATE)
    plot_spectrogram(
        spec,
        sr=SAMPLE_RATE,
        title="Phase 4 — Mel Spectrogram (fully cleaned audio)",
        save_path=os.path.join(OUT_DIR, "test_phase4_spectrogram.png"),
    )


if __name__ == "__main__":
    main()

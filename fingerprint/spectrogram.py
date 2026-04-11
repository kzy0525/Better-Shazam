"""
fingerprint/spectrogram.py — Spectrogram pipeline.

Converts a raw audio array into a mel spectrogram and extracts sparse,
consistent peak points that serve as the input to Phase 3 hashing.
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from typing import List, Tuple

from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, FMIN, FMAX


def mel_frequencies() -> np.ndarray:
    """Return the centre frequency (Hz) for each of the N_MELS mel bins.

    Computed directly from the mel scale formula — no librosa dependency
    at call time so this is safe to use in test scripts.

    Returns:
        1-D float64 array of shape (N_MELS,).
    """
    fmin_mel = 2595.0 * np.log10(1.0 + FMIN / 700.0)
    fmax_mel = 2595.0 * np.log10(1.0 + FMAX / 700.0)
    mels = np.linspace(fmin_mel, fmax_mel, N_MELS)
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def compute_mel_spectrogram(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Compute a mel spectrogram from a raw audio array.

    Pipeline:
        1. STFT (n_fft=2048, hop_length=512)
        2. Map to mel filterbank (128 bins, 80–8000 Hz)
        3. Convert power to decibels (ref=np.max → range roughly -80 to 0 dB)

    Args:
        audio: 1-D float32 numpy array (mono).
        sr:    Sample rate of the input in Hz.

    Returns:
        2-D float32 numpy array of shape (n_mels, time_frames),
        values in dB relative to the peak power in the clip.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def extract_peaks(
    spectrogram: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_peaks_per_sec: int = 20,
    min_amplitude_db: float = -60.0,
    neighborhood: int = 20,
    freq_ceiling_bin: int = 100,
    time_border: int = 5,
) -> List[Tuple[int, int, float]]:
    """Extract sparse local-maxima peaks from a mel spectrogram.

    Uses a maximum filter over a (neighborhood × neighborhood) window to
    find local maxima, then keeps only the strongest n_peaks_per_sec peaks
    per second of audio. Result is deterministic for a given input.

    Args:
        spectrogram:      2-D float32 array (n_mels × time_frames) in dB.
        sr:               Sample rate used to compute the spectrogram.
        n_peaks_per_sec:  Maximum peaks to keep per second of audio.
        min_amplitude_db: Discard peaks below this dB threshold.
        neighborhood:     Side length of the maximum-filter window (pixels).

        freq_ceiling_bin: Ignore peaks at or above this mel bin index (0-indexed).
                          Cuts high-frequency noise above the useful range.
        time_border:      Number of frames to ignore at each end of the spectrogram
                          to avoid STFT boundary artifacts.

    Returns:
        List of (time_frame, freq_bin, amplitude_db) tuples sorted by
        descending amplitude. time_frame values are relative to the full
        spectrogram (boundary frames added back in).
    """
    # Slice out boundary frames and high-frequency bins before peak search
    interior = spectrogram[:freq_ceiling_bin, time_border:-time_border]

    # Local maximum filter — a point is a peak if it equals the neighbourhood max
    local_max = maximum_filter(interior, size=(neighborhood, neighborhood))
    is_peak = (interior == local_max) & (interior >= min_amplitude_db)

    freq_bins, time_frames_interior = np.where(is_peak)
    amplitudes = interior[freq_bins, time_frames_interior]

    # Restore absolute time frame indices (offset by the trimmed border)
    time_frames = time_frames_interior + time_border

    # Bundle into tuples and sort strongest first (deterministic for equal amps
    # because we use a stable sort on a derived key)
    peaks = sorted(
        zip(time_frames.tolist(), freq_bins.tolist(), amplitudes.tolist()),  # type: ignore[union-attr]
        key=lambda p: p[2],
        reverse=True,
    )

    # Thin to n_peaks_per_sec per second of audio
    duration_sec = (spectrogram.shape[1] * HOP_LENGTH) / sr
    max_peaks = int(n_peaks_per_sec * duration_sec)
    peaks = peaks[:max_peaks]

    return peaks


def plot_spectrogram(
    spectrogram: np.ndarray,
    sr: int = SAMPLE_RATE,
    peaks: List[Tuple[int, int, float]] | None = None,
    title: str = "Mel Spectrogram",
    save_path: str | None = None,
) -> None:
    """Plot a mel spectrogram heatmap, optionally overlaying peak points.

    Args:
        spectrogram: 2-D float32 array (n_mels × time_frames) in dB.
        sr:          Sample rate used to compute the spectrogram.
        peaks:       Optional list of (time_frame, freq_bin, amplitude_db)
                     tuples from extract_peaks. Drawn as scatter points.
        title:       Figure title string.
        save_path:   If provided, save the figure to this file path.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    img = librosa.display.specshow(
        spectrogram,
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
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    if peaks:
        # Convert (time_frame, freq_bin) back to axis coordinates
        times = librosa.frames_to_time(
            [p[0] for p in peaks], sr=sr, hop_length=HOP_LENGTH
        )
        freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
        freq_vals = [freqs[p[1]] for p in peaks]

        ax.scatter(
            times,
            freq_vals,
            color="cyan",
            s=8,
            linewidths=0,
            alpha=0.75,
            label=f"{len(peaks)} peaks",
        )
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")

    plt.show()

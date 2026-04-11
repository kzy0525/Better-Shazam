"""
audio/preprocess.py — Audio preprocessing pipeline.

Phase 1: bandpass_filter        — Butterworth bandpass (80–8000 Hz)
Phase 4: separate_stems         — Demucs htdemucs source separation
         wiener_filter          — Wiener filter for residual stationary noise
         full_preprocess_pipeline — single entry point chaining all steps
"""

import numpy as np
import torch
from scipy.signal import butter, sosfilt, wiener, resample_poly
from demucs.apply import apply_model

from config import SAMPLE_RATE, FMIN, FMAX

# Demucs model is loaded once and cached at module level to avoid
# reloading the 80 MB checkpoint on every call.
_demucs_model = None
_DEMUCS_SR = 44100  # htdemucs internal sample rate


def _get_demucs_model():
    """Load and cache the htdemucs model (downloaded on first call)."""
    global _demucs_model
    if _demucs_model is None:
        from demucs.pretrained import get_model
        _demucs_model = get_model("htdemucs")
        _demucs_model.eval()
    return _demucs_model


# ---------------------------------------------------------------------------
# Stage 1 — Demucs source separation
# ---------------------------------------------------------------------------

def separate_stems(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Separate audio into musical stems and recombine, discarding non-music.

    Uses the htdemucs pretrained model (drums, bass, other, vocals) to
    perform source separation. All four stems are summed back into a single
    array — the separation process itself removes dynamic background noise
    (crowd, talking, environmental sounds) that doesn't fit any stem.

    Internally upsamples to 44100 Hz (demucs requirement) then downsamples
    back to sample_rate after processing. Input and output shapes match.

    Args:
        audio:       1-D float32 numpy array (mono).
        sample_rate: Sample rate of the input in Hz.

    Returns:
        Reconstructed audio array, same dtype and shape as input.
    """
    model = _get_demucs_model()

    # Upsample to demucs native rate (22050 → 44100, ratio 2/1)
    up_ratio = _DEMUCS_SR // sample_rate
    audio_up = resample_poly(audio, up_ratio, 1).astype(np.float32)

    # Demucs expects (batch=1, channels=2, samples) — duplicate mono to stereo
    stereo = np.stack([audio_up, audio_up], axis=0)              # (2, samples)
    tensor = torch.from_numpy(stereo).unsqueeze(0)               # (1, 2, samples)

    with torch.no_grad():
        stems = apply_model(model, tensor)  # (1, n_stems=4, 2, samples)

    # Sum all stems: drums + bass + other + vocals → full reconstructed signal
    combined = stems[0].sum(dim=0)        # (2, samples) — stereo
    mono_up = combined.mean(dim=0).numpy()  # (samples,)  — back to mono

    # Downsample back to original sample rate
    audio_out = resample_poly(mono_up, 1, up_ratio).astype(np.float32)

    # Trim/pad to exact original length (resample_poly can introduce ±1 sample)
    audio_out = audio_out[:len(audio)]
    if len(audio_out) < len(audio):
        audio_out = np.pad(audio_out, (0, len(audio) - len(audio_out)))

    return audio_out


# ---------------------------------------------------------------------------
# Stage 2 — Wiener filter
# ---------------------------------------------------------------------------

def wiener_filter(
    audio: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """Apply a Wiener filter to suppress residual stationary noise.

    Handles hiss, electrical hum, and other stationary artefacts left
    over after Demucs stem separation. Uses scipy.signal.wiener with a
    1-D sliding window.

    Args:
        audio:       1-D float32 numpy array (mono).
        window_size: Length of the Wiener filter window in samples.

    Returns:
        Filtered audio array, same dtype and shape as input.
    """
    filtered = wiener(audio, mysize=window_size)
    return filtered.astype(audio.dtype)


# ---------------------------------------------------------------------------
# Phase 1 — Bandpass filter
# ---------------------------------------------------------------------------

def bandpass_filter(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    low_hz: float = FMIN,
    high_hz: float = FMAX,
    order: int = 5,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter to an audio array.

    Uses second-order sections (SOS) representation for numerical
    stability compared to the standard transfer-function form.

    Args:
        audio:       1-D float32 numpy array (mono).
        sample_rate: Sample rate of the input in Hz.
        low_hz:      Lower cutoff frequency in Hz.
        high_hz:     Upper cutoff frequency in Hz.
        order:       Filter order. Higher = steeper rolloff, more phase delay.

    Returns:
        Filtered audio array, same dtype and shape as input.
    """
    nyquist = sample_rate / 2.0
    low = low_hz / nyquist
    high = high_hz / nyquist

    sos = butter(order, [low, high], btype="band", output="sos")
    filtered = sosfilt(sos, audio)

    return filtered.astype(audio.dtype)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def full_preprocess_pipeline(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Run the complete preprocessing pipeline on a raw audio array.

    Steps (in order):
        1. Demucs source separation — removes dynamic background noise
        2. Wiener filter — removes residual stationary noise/hiss
        3. Butterworth bandpass filter (FMIN–FMAX Hz)

    This is the single entry point all phases should call. Using the same
    pipeline for both registration and query is critical — fingerprints must
    be generated under identical conditions to match.

    Args:
        audio:       1-D float32 numpy array (mono).
        sample_rate: Sample rate in Hz.

    Returns:
        Fully preprocessed 1-D float32 numpy array, same length as input.
    """
    print("  [1/3] Demucs source separation...")
    audio = separate_stems(audio, sample_rate)
    print("  [2/3] Wiener filter...")
    audio = wiener_filter(audio)
    print("  [3/3] Bandpass filter...")
    audio = bandpass_filter(audio, sample_rate)
    return audio


def preprocess(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Backwards-compatible alias for full_preprocess_pipeline."""
    return full_preprocess_pipeline(audio, sample_rate)

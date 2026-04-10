"""
audio/preprocess.py — Audio preprocessing pipeline.

Applies a Butterworth bandpass filter (80 Hz–8000 Hz) to a raw audio
array using scipy. Additional steps (noise reduction, normalisation)
are isolated so they can be tested or skipped independently.
"""

import numpy as np
from scipy.signal import butter, sosfilt

from config import SAMPLE_RATE, FMIN, FMAX


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


def preprocess(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Run the Phase 1 preprocessing pipeline on a raw audio array.

    Steps:
        1. Bandpass filter (FMIN–FMAX Hz via Butterworth)

    Args:
        audio:       1-D float32 numpy array (mono).
        sample_rate: Sample rate of the input in Hz.

    Returns:
        Preprocessed 1-D float32 numpy array, same length as input.
    """
    audio = bandpass_filter(audio, sample_rate)
    return audio

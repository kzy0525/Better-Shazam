"""
audio/capture.py — Microphone recording.

Records a fixed-length audio clip from the default input device at
SAMPLE_RATE Hz mono and returns it as a float32 numpy array.
Optionally saves the result to a WAV file.
"""

import numpy as np
import sounddevice as sd
import soundfile as sf

from config import SAMPLE_RATE, CLIP_LENGTH
from audio.preprocess import full_preprocess_pipeline


def record(
    duration: float = CLIP_LENGTH,
    sample_rate: int = SAMPLE_RATE,
    save_path: str | None = None,
) -> np.ndarray:
    """Record audio from the default microphone.

    Args:
        duration:    Length of the recording in seconds.
        sample_rate: Sample rate in Hz. Defaults to 22050.
        save_path:   If provided, write the clip to this WAV file path.

    Returns:
        1-D float32 numpy array of shape (duration * sample_rate,),
        values in [-1.0, 1.0].
    """
    print(f"Recording {duration}s at {sample_rate} Hz...")

    audio = sd.rec(
        frames=int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()  # block until recording is complete

    # sd.rec returns shape (frames, channels) — squeeze to 1-D
    audio = audio.squeeze()

    print("Recording complete.")

    if save_path is not None:
        sf.write(save_path, audio, sample_rate)
        print(f"Saved raw recording to: {save_path}")

    return audio


def record_and_preprocess(
    duration: float = CLIP_LENGTH,
    sample_rate: int = SAMPLE_RATE,
    save_path: str | None = None,
) -> np.ndarray:
    """Record from the microphone and immediately run the full preprocessing pipeline.

    Chains record() → full_preprocess_pipeline() in one call. This is the
    single entry point for all audio capture going forward — use this instead
    of calling record() and preprocess steps separately.

    Args:
        duration:    Length of the recording in seconds.
        sample_rate: Sample rate in Hz.
        save_path:   If provided, save the cleaned audio to this WAV file path.

    Returns:
        Fully preprocessed 1-D float32 numpy array (Demucs separated + Wiener filtered + bandpass filtered).
    """
    raw = record(duration=duration, sample_rate=sample_rate)
    cleaned = full_preprocess_pipeline(raw, sample_rate=sample_rate)

    if save_path is not None:
        sf.write(save_path, cleaned, sample_rate)
        print(f"Saved preprocessed recording to: {save_path}")

    return cleaned

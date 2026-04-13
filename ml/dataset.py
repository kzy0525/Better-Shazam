"""
ml/dataset.py — Triplet dataset construction for AudioEmbedder training.

Slices WAV files into fixed-length clips, applies audiomentations
augmentations (including TimeStretch for tempo robustness), and
assembles (anchor, positive, negative) triplets as mel spectrograms
ready to feed into the CNN.

Supports two data sources:
  - load_gtzan()         — 1000-file GTZAN dataset (preferred for training)
  - build_triplet_dataset() — accepts GTZAN tuples or falls back to a flat
                              songs directory for quick local testing
"""

import os
import random
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
import audiomentations as A
import matplotlib.pyplot as plt

from fingerprint.spectrogram import compute_mel_spectrogram
from config import SAMPLE_RATE


# ---------------------------------------------------------------------------
# GTZAN loader
# ---------------------------------------------------------------------------

GTZAN_GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]


def load_gtzan(
    gtzan_dir: str = "data/genres_original/",
    genres: List[str] | None = None,
) -> List[Tuple[str, str]]:
    """Scan the GTZAN dataset directory and return all (filepath, genre) tuples.

    Expects the standard GTZAN layout:
        gtzan_dir/
            blues/       blues.00000.wav ... blues.00099.wav
            classical/   classical.00000.wav ...
            ...          (10 genres × 100 files = 1000 total)

    Args:
        gtzan_dir: Path to the genres_original/ folder.
        genres:    Optional list of genre names to include. If None, all 10
                   genres are loaded. E.g. ["blues", "classical", "rock"].

    Returns:
        List of (filepath, genre_label) tuples for all found WAV files.
        genre_label is the subfolder name (e.g. "blues", "classical").
    """
    if not os.path.isdir(gtzan_dir):
        raise FileNotFoundError(
            f"GTZAN directory not found: {gtzan_dir}\n"
            "Download from: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification"
        )

    active_genres = genres if genres is not None else GTZAN_GENRES
    entries: List[Tuple[str, str]] = []
    counts: defaultdict[str, int] = defaultdict(int)

    for genre in active_genres:
        genre_dir = os.path.join(gtzan_dir, genre)
        if not os.path.isdir(genre_dir):
            print(f"  Warning: genre folder not found — {genre_dir}")
            continue
        for fname in sorted(os.listdir(genre_dir)):
            if fname.lower().endswith(".wav"):
                entries.append((os.path.join(genre_dir, fname), genre))
                counts[genre] += 1

    print(f"GTZAN loaded — {len(entries)} files across {len(counts)} genres:")
    for genre in active_genres:
        if genre in counts:
            print(f"  {genre:<12} {counts[genre]} files")

    return entries


# ---------------------------------------------------------------------------
# Audio slicing
# ---------------------------------------------------------------------------

def slice_audio(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    clip_length: float = 5.0,
) -> List[np.ndarray]:
    """Slice a numpy audio array into non-overlapping fixed-length clips.

    Args:
        audio:       1-D float32 numpy array (mono).
        sr:          Sample rate in Hz.
        clip_length: Desired clip length in seconds.

    Returns:
        List of 1-D float32 numpy arrays, each exactly clip_length * sr samples.
        The last clip is discarded if it would be shorter than clip_length.
    """
    clip_samples = int(clip_length * sr)
    clips = []
    for start in range(0, len(audio) - clip_samples + 1, clip_samples):
        clips.append(audio[start : start + clip_samples].copy())
    return clips


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def _build_augmenter(sr: int = SAMPLE_RATE) -> A.Compose:
    """Build the audiomentations augmentation composition."""
    return A.Compose([
        A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        A.TimeStretch(min_rate=0.80, max_rate=1.20, p=0.8),
        A.PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
        A.RoomSimulator(p=0.4),
        A.LowPassFilter(min_cutoff_freq=4000, max_cutoff_freq=8000, p=0.3),
    ])


def augment_clip(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Apply random augmentations to a single audio clip.

    TimeStretch can change the clip length by up to ±20%. After augmentation
    the clip is trimmed or zero-padded back to the original length so all
    tensors remain the same shape.

    Args:
        audio: 1-D float32 numpy array (mono), exactly clip_length * sr samples.
        sr:    Sample rate in Hz.

    Returns:
        Augmented 1-D float32 numpy array, same length as input.
    """
    original_len = len(audio)
    augmenter = _build_augmenter(sr)
    augmented = augmenter(samples=audio, sample_rate=sr)

    # Restore original length after TimeStretch
    if len(augmented) > original_len:
        augmented = augmented[:original_len]
    elif len(augmented) < original_len:
        augmented = np.pad(augmented, (0, original_len - len(augmented)))

    # RoomSimulator can produce NaN/inf via fft convolution on bad RIRs —
    # replace with the clean input so the clip is still usable
    if not np.isfinite(augmented).all():
        augmented = audio.copy()

    return augmented.astype(np.float32)


# ---------------------------------------------------------------------------
# Spectrogram helpers
# ---------------------------------------------------------------------------

def _to_spectrogram(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Convert audio to a normalised mel spectrogram (zero mean, unit variance)."""
    spec = compute_mel_spectrogram(audio, sr=sr)
    mean, std = spec.mean(), spec.std()
    if std > 0:
        spec = (spec - mean) / std
    return spec  # shape: (N_MELS, time_frames)


def _spec_to_tensor(spec: np.ndarray) -> torch.Tensor:
    """Add channel dim and convert to float32 tensor: (1, N_MELS, time_frames)."""
    return torch.from_numpy(spec).unsqueeze(0).float()


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class TripletAudioDataset(Dataset):
    """PyTorch Dataset yielding (anchor, positive, negative) spectrogram triplets.

    Each triplet:
        anchor   — clean mel spectrogram of a clip from Song A
        positive — augmented (possibly tempo-shifted) version of the same clip
        negative — augmented clip from a different song

    Tensors are shape (1, N_MELS, time_frames).
    """

    def __init__(self, triplets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
        self.triplets = triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        anchor, positive, negative = self.triplets[idx]
        return (
            _spec_to_tensor(anchor),
            _spec_to_tensor(positive),
            _spec_to_tensor(negative),
        )


def build_triplet_dataset(
    source: List[Tuple[str, str]] | str = "songs/",
    clips_per_song: int = 10,
    augmentations_per_clip: int = 5,
) -> TripletAudioDataset:
    """Build a triplet dataset from a list of (filepath, label) tuples or a flat directory.

    Accepts either:
      - A list of (filepath, genre_label) tuples from load_gtzan() — preferred
      - A flat directory path (str) of WAV files — for quick local testing

    For each song:
        - Slices into clips (up to clips_per_song, evenly spaced)
        - For each clip generates augmentations_per_clip augmented versions
        - Anchor: clean mel spectrogram of the clip
        - Positive: augmented version of the same clip (may be tempo-shifted)
        - Negative: augmented clip from a different filepath; preferably from
          a different genre/label to make negatives harder and improve training

    Args:
        source:                List of (filepath, label) tuples, or a directory path string.
        clips_per_song:        Maximum clips to take per file (evenly spaced).
        augmentations_per_clip: Number of positive augmentations per clip.

    Returns:
        TripletAudioDataset ready for use with a PyTorch DataLoader.
    """
    # Resolve source into a list of (filepath, label) tuples
    if isinstance(source, str):
        songs_dir = source
        wav_files = sorted([
            os.path.join(songs_dir, f)
            for f in os.listdir(songs_dir)
            if f.lower().endswith(".wav")
        ])
        if len(wav_files) < 2:
            raise ValueError(f"Need at least 2 WAV files in {songs_dir}, found {len(wav_files)}")
        entries = [(path, "unknown") for path in wav_files]
    else:
        entries = source

    if len(entries) < 2:
        raise ValueError(f"Need at least 2 entries, got {len(entries)}")

    # Group by label so we can find cross-genre negatives
    label_to_indices: defaultdict[str, List[int]] = defaultdict(list)
    for i, (_, label) in enumerate(entries):
        label_to_indices[label].append(i)

    # Load clips indexed by entry index
    print(f"Loading clips from {len(entries)} files...")
    index_to_clips: dict[int, List[np.ndarray]] = {}
    for i, (path, _) in enumerate(entries):
        try:
            audio, sr = sf.read(path, dtype="float32")
        except Exception as e:
            print(f"  Skipping {path}: {e}")
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        clips = slice_audio(audio, sr=sr)
        if not clips:
            continue
        if len(clips) > clips_per_song:
            indices = np.linspace(0, len(clips) - 1, clips_per_song, dtype=int)
            clips = [clips[j] for j in indices]
        index_to_clips[i] = clips

    valid_indices = list(index_to_clips.keys())
    if len(valid_indices) < 2:
        raise ValueError("Need at least 2 files with extractable clips.")

    # Build triplets
    triplets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for anchor_idx in valid_indices:
        _, anchor_label = entries[anchor_idx]
        anchor_clips = index_to_clips[anchor_idx]

        # Prefer negatives from a different genre; fall back to any other file
        different_genre = [
            j for j in valid_indices
            if j != anchor_idx and entries[j][1] != anchor_label
        ]
        same_genre_other = [
            j for j in valid_indices
            if j != anchor_idx and entries[j][1] == anchor_label
        ]
        neg_pool = different_genre if different_genre else same_genre_other

        for clip in anchor_clips:
            anchor_spec = _to_spectrogram(clip)
            for _ in range(augmentations_per_clip):
                pos_audio = augment_clip(clip)
                positive_spec = _to_spectrogram(pos_audio)

                neg_idx = random.choice(neg_pool)
                neg_clip = random.choice(index_to_clips[neg_idx])
                neg_audio = augment_clip(neg_clip)
                negative_spec = _to_spectrogram(neg_audio)

                triplets.append((anchor_spec, positive_spec, negative_spec))

    random.shuffle(triplets)
    print(f"Built {len(triplets)} triplets from {len(valid_indices)} files.")
    return TripletAudioDataset(triplets)


# ---------------------------------------------------------------------------
# Verification utility
# ---------------------------------------------------------------------------

def verify_dataset(
    dataset: TripletAudioDataset,
    n_samples: int = 5,
    save_path: str = "ml/dataset_verify.png",
) -> None:
    """Plot n_samples triplets side by side to visually confirm augmentation.

    Args:
        dataset:    A TripletAudioDataset instance.
        n_samples:  Number of triplets to visualise.
        save_path:  File path to save the figure.
    """
    n_samples = min(n_samples, len(dataset))
    fig, axes = plt.subplots(n_samples, 3, figsize=(14, 3 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    labels = [
        "Anchor (clean)",
        "Positive (augmented\n— may be different tempo)",
        "Negative (different song)",
    ]
    colors = ["Blues", "Oranges", "Purples"]

    for row, idx in enumerate(random.sample(range(len(dataset)), n_samples)):
        anchor, positive, negative = dataset[idx]
        specs = [anchor.squeeze().numpy(), positive.squeeze().numpy(), negative.squeeze().numpy()]
        for col, (spec, label, cmap) in enumerate(zip(specs, labels, colors)):
            ax = axes[row, col]
            ax.imshow(spec, aspect="auto", origin="lower", cmap=cmap)
            if row == 0:
                ax.set_title(label, fontsize=9, fontweight="bold")
            ax.set_xlabel("Time frames")
            ax.set_ylabel("Mel bin")

    fig.suptitle("Dataset Verification — Triplet Samples", fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=130)
    print(f"Dataset verification plot saved to: {save_path}")
    plt.show()

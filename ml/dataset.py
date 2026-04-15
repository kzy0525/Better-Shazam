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
import torch.nn.functional as F
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
    n_clips: int | None = None,
    debug_name: str | None = None,
) -> List[np.ndarray]:
    """Slice a numpy audio array into fixed-length clips, evenly spaced across the song.

    When n_clips is specified and the song is long enough, clips are sampled
    evenly across the full duration (intro, verse, chorus, bridge, outro) rather
    than sequentially from the start. If the song is shorter than
    n_clips * clip_length seconds, all available non-overlapping clips are returned.

    Args:
        audio:       1-D float32 numpy array (mono).
        sr:          Sample rate in Hz.
        clip_length: Desired clip length in seconds.
        n_clips:     Number of evenly spaced clips to return. If None, all
                     non-overlapping clips are returned.
        debug_name:  If provided, prints the selected clip start times in
                     seconds for verification.

    Returns:
        List of 1-D float32 numpy arrays, each exactly clip_length * sr samples.
    """
    clip_samples = int(clip_length * sr)
    total_available = len(audio) // clip_samples

    if total_available == 0:
        return []

    if n_clips is None or total_available <= n_clips:
        # Return all non-overlapping clips sequentially
        indices = list(range(total_available))
    else:
        # Evenly space n_clips across all available clip positions
        indices = [int(x) for x in np.linspace(0, total_available - 1, n_clips)]

    clips = []
    for idx in indices:
        start = idx * clip_samples
        clips.append(audio[start : start + clip_samples].copy())

    if debug_name is not None:
        times = [f"{idx * clip_length:.1f}s" for idx in indices]
        print(f"  Clip start times for {debug_name}: [{', '.join(times)}]")

    return clips


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def _build_augmenter(sr: int = SAMPLE_RATE) -> A.Compose:
    """Build the audiomentations augmentation composition."""
    return A.Compose([
        A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        A.AddColorNoise(p=0.4),
        A.TimeStretch(min_rate=0.80, max_rate=1.20, p=0.8),
        A.PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
        A.RoomSimulator(p=0.6),
        A.LowPassFilter(min_cutoff_freq=4000, max_cutoff_freq=8000, p=0.3),
        A.TanhDistortion(p=0.3),
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
    gtzan_source: List[Tuple[str, str]] | None = None,
    songs_dir: str | None = None,
    gtzan_clips_per_file: int = 3,
    gtzan_augmentations_per_clip: int = 3,
    songs_clips_per_file: int = 10,
    songs_augmentations_per_clip: int = 20,
) -> TripletAudioDataset:
    """Build a triplet dataset from GTZAN entries and/or a local songs directory.

    Handles two data sources with different sampling rates:
      - gtzan_source: List of (filepath, genre_label) tuples from load_gtzan().
                      Uses gtzan_clips_per_file / gtzan_augmentations_per_clip.
      - songs_dir:    Flat directory of WAV files (the user's own song library).
                      Uses songs_clips_per_file / songs_augmentations_per_clip.

    Negative selection:
      - GTZAN anchor  → prefers a different-genre GTZAN file; falls back to
                        same-genre-other GTZAN, then songs_dir files.
      - songs/ anchor → always prefers another songs_dir file; falls back to
                        GTZAN files only if no other songs/ file is available.

    Prints a triplet count breakdown at the end showing GTZAN vs songs/
    contributions separately.

    Args:
        gtzan_source:                 (filepath, genre_label) tuples from load_gtzan().
        songs_dir:                    Directory of WAV files for the user's song library.
        gtzan_clips_per_file:         Max clips per GTZAN file (evenly spaced).
        gtzan_augmentations_per_clip: Augmented positives per GTZAN clip.
        songs_clips_per_file:         Max clips per songs/ file.
        songs_augmentations_per_clip: Augmented positives per songs/ clip.

    Returns:
        TripletAudioDataset ready for use with a PyTorch DataLoader.
    """
    if gtzan_source is None and songs_dir is None:
        raise ValueError("Provide at least one of gtzan_source or songs_dir.")

    # Build a unified tagged entry list: (filepath, label, source)
    # source is "gtzan" or "songs"
    tagged: List[Tuple[str, str, str]] = []

    if gtzan_source:
        for path, label in gtzan_source:
            tagged.append((path, label, "gtzan"))

    songs_paths: List[str] = []
    if songs_dir and os.path.isdir(songs_dir):
        songs_paths = sorted([
            os.path.join(songs_dir, f)
            for f in os.listdir(songs_dir)
            if f.lower().endswith(".wav")
        ])
        for path in songs_paths:
            tagged.append((path, "songs", "songs"))

    if len(tagged) < 2:
        raise ValueError(f"Need at least 2 files total, got {len(tagged)}.")

    # Load clips for each entry, applying per-source limits
    n_gtzan = sum(1 for _, _, s in tagged if s == "gtzan")
    n_songs = sum(1 for _, _, s in tagged if s == "songs")
    print(f"Loading clips from {len(tagged)} files ({n_gtzan} GTZAN, {n_songs} songs/)...")

    index_to_clips: dict[int, List[np.ndarray]] = {}
    first_processed = True
    for i, (path, _, source) in enumerate(tagged):
        limit = gtzan_clips_per_file if source == "gtzan" else songs_clips_per_file
        try:
            audio, sr = sf.read(path, dtype="float32")
        except Exception as e:
            print(f"  Skipping {path}: {e}")
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        name = os.path.splitext(os.path.basename(path))[0] if first_processed else None
        clips = slice_audio(audio, sr=sr, n_clips=limit, debug_name=name)
        first_processed = False
        if not clips:
            continue
        index_to_clips[i] = clips

    valid_indices = list(index_to_clips.keys())
    if len(valid_indices) < 2:
        raise ValueError("Need at least 2 files with extractable clips.")

    # Index sets by source for negative selection
    gtzan_indices = [i for i in valid_indices if tagged[i][2] == "gtzan"]
    songs_indices  = [i for i in valid_indices if tagged[i][2] == "songs"]

    # Build triplets
    triplets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    gtzan_triplet_count = 0
    songs_triplet_count = 0

    for anchor_idx in valid_indices:
        _, anchor_label, anchor_source = tagged[anchor_idx]
        anchor_clips = index_to_clips[anchor_idx]
        aug_count = (gtzan_augmentations_per_clip if anchor_source == "gtzan"
                     else songs_augmentations_per_clip)

        # Negative pool selection
        if anchor_source == "songs":
            # Prefer another songs/ file; fall back to GTZAN
            other_songs = [j for j in songs_indices if j != anchor_idx]
            neg_pool = other_songs if other_songs else gtzan_indices
        else:
            # GTZAN anchor: prefer different-genre GTZAN, then same-genre-other, then songs/
            diff_genre = [
                j for j in gtzan_indices
                if j != anchor_idx and tagged[j][1] != anchor_label
            ]
            same_genre_other = [
                j for j in gtzan_indices
                if j != anchor_idx and tagged[j][1] == anchor_label
            ]
            neg_pool = diff_genre or same_genre_other or songs_indices

        for clip in anchor_clips:
            anchor_spec = _to_spectrogram(clip)
            for _ in range(aug_count):
                pos_audio = augment_clip(augment_clip(clip))
                positive_spec = _to_spectrogram(pos_audio)

                neg_idx = random.choice(neg_pool)
                neg_clip = random.choice(index_to_clips[neg_idx])
                neg_audio = augment_clip(neg_clip)
                negative_spec = _to_spectrogram(neg_audio)

                triplets.append((anchor_spec, positive_spec, negative_spec))

                if anchor_source == "gtzan":
                    gtzan_triplet_count += 1
                else:
                    songs_triplet_count += 1

    random.shuffle(triplets)

    print(f"\nTriplet breakdown:")
    print(f"  GTZAN anchors : {gtzan_triplet_count} triplets  "
          f"({len(gtzan_indices)} files × up to {gtzan_clips_per_file} clips "
          f"× {gtzan_augmentations_per_clip} aug)")
    print(f"  songs/ anchors: {songs_triplet_count} triplets  "
          f"({len(songs_indices)} files × up to {songs_clips_per_file} clips "
          f"× {songs_augmentations_per_clip} aug)")
    print(f"  Total         : {len(triplets)} triplets")

    return TripletAudioDataset(triplets)


# ---------------------------------------------------------------------------
# Spectrogram cache
# ---------------------------------------------------------------------------

def build_spectrogram_cache(
    songs_dir: str,
    cache_dir: str = "ml/spectrogram_cache",
    clip_length: float = 5.0,
    n_clips: int = 10,
) -> List[Tuple[str, int, str]]:
    """Preprocess and cache mel spectrograms for all songs/ clips.

    Runs the full preprocessing pipeline (Demucs → Wiener → bandpass →
    mel spectrogram) on each clip once and saves each result as a .npy
    file. Skips clips whose cache file already exists. This avoids
    re-running Demucs on every training run.

    Args:
        songs_dir:   Directory of WAV files (22050 Hz mono).
        cache_dir:   Directory to store .npy spectrogram files.
        clip_length: Clip duration in seconds (must match training).
        n_clips:     Number of evenly spaced clips per song.

    Returns:
        List of (song_name, clip_index, cache_path) tuples for all cached clips.
    """
    # Import here to avoid circular imports at module level
    from audio.preprocess import full_preprocess_pipeline

    os.makedirs(cache_dir, exist_ok=True)

    wav_files = sorted([
        os.path.join(songs_dir, f)
        for f in os.listdir(songs_dir)
        if f.lower().endswith(".wav")
    ])

    entries = []   # (song_name, clip_idx, cache_path)
    n_cached = 0
    n_skipped = 0

    print(f"Building spectrogram cache for {len(wav_files)} songs → {cache_dir}")

    for wav_path in wav_files:
        song_name = os.path.splitext(os.path.basename(wav_path))[0]

        try:
            audio, sr = sf.read(wav_path, dtype="float32")
        except Exception as e:
            print(f"  Skipping {wav_path}: {e}")
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        clips = slice_audio(audio, sr=sr, clip_length=clip_length, n_clips=n_clips)

        for clip_idx, clip in enumerate(clips):
            cache_path = os.path.join(cache_dir, f"{song_name}_{clip_idx}.npy")
            entries.append((song_name, clip_idx, cache_path))

            if os.path.exists(cache_path):
                n_skipped += 1
                continue

            # Full preprocessing on the raw clip, then spectrogram.
            # Wiener filter can produce NaN/inf on short clips — fall back to raw.
            processed = full_preprocess_pipeline(clip, sample_rate=sr)
            if not np.isfinite(processed).all():
                processed = clip.copy()
            spec = _to_spectrogram(processed, sr)
            np.save(cache_path, spec)
            n_cached += 1

    print(f"  Cached: {n_cached} new clips, {n_skipped} already existed "
          f"({n_cached + n_skipped} total)")
    return entries


# ---------------------------------------------------------------------------
# Semi-hard negative mining
# ---------------------------------------------------------------------------

def mine_triplets(
    cache_entries: List[Tuple[str, int, str]],
    model,
    margin: float = 0.3,
    augmentations_per_clip: int = 20,
    epoch: int = 1,
) -> TripletAudioDataset:
    """Build a triplet dataset using semi-hard negative mining.

    At the start of each epoch, embeds all cached spectrograms with the
    current model, then for each anchor finds:
      - A positive: an augmented version of the same clip
      - A semi-hard negative: a clip from a different song where
            d(a, p) < d(a, n) < d(a, p) + margin
        Falls back to the hardest available negative (smallest d(a, n))
        if no semi-hard negative exists.

    Args:
        cache_entries:          List of (song_name, clip_idx, cache_path) from
                                build_spectrogram_cache().
        model:                  AudioEmbedder in any training state — will be
                                temporarily set to eval() for embedding, then
                                restored to train() after.
        margin:                 Triplet loss margin (must match training margin).
        augmentations_per_clip: Number of augmented positives per anchor clip.
        epoch:                  Current epoch number, used for logging only.

    Returns:
        TripletAudioDataset with freshly mined triplets.
    """
    was_training = model.training
    model.eval()

    # Embed all cached clips
    all_specs: List[np.ndarray] = []
    all_names: List[str] = []
    valid_entries: List[Tuple[str, int, str]] = []

    for song_name, clip_idx, cache_path in cache_entries:
        if not os.path.exists(cache_path):
            continue
        spec = np.load(cache_path)
        all_specs.append(spec)
        all_names.append(song_name)
        valid_entries.append((song_name, clip_idx, cache_path))

    if len(valid_entries) < 2:
        raise ValueError("Need at least 2 cached clips for mining.")

    with torch.no_grad():
        tensors = torch.stack([_spec_to_tensor(s) for s in all_specs])  # (N, 1, H, W)
        embeddings = model(tensors)                                       # (N, 128)

    # Per-song index sets
    song_to_indices: dict[str, List[int]] = defaultdict(list)
    for i, name in enumerate(all_names):
        song_to_indices[name].append(i)

    triplets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    neg_distances: List[float] = []

    for anchor_i, (song_name, _, cache_path) in enumerate(valid_entries):
        anchor_emb = embeddings[anchor_i]
        anchor_spec = all_specs[anchor_i]

        # Indices belonging to a different song
        neg_indices = [j for j in range(len(valid_entries)) if all_names[j] != song_name]
        if not neg_indices:
            continue

        # d_pos is estimated as 0 — the augmented positive is derived from the same
        # clip so its true distance is small. Semi-hard condition becomes:
        # 0 < d(a, n) < margin, i.e. negatives within the margin.
        d_pos = 0.0

        neg_embs = embeddings[neg_indices]
        dists = F.pairwise_distance(anchor_emb.unsqueeze(0).expand_as(neg_embs), neg_embs)
        dists_np = dists.cpu().numpy()

        # Semi-hard: d_pos < d_neg < d_pos + margin
        semi_hard_mask = (dists_np > d_pos) & (dists_np < d_pos + margin)
        if semi_hard_mask.any():
            # Pick randomly among semi-hard negatives
            candidates = [neg_indices[k] for k in np.where(semi_hard_mask)[0]]
            neg_i = random.choice(candidates)
        else:
            # Fall back to hardest negative (smallest distance)
            neg_i = neg_indices[int(np.argmin(dists_np))]

        neg_distances.append(float(dists_np[neg_indices.index(neg_i)]))
        neg_spec = all_specs[neg_i]

        # Generate augmented positives
        for _ in range(augmentations_per_clip):
            # Augment from the raw WAV clip: reload and augment
            pos_raw = anchor_spec.flatten()[:int(5.0 * SAMPLE_RATE)].astype(np.float32)
            pos_aug = augment_clip(augment_clip(pos_raw))
            pos_spec = _to_spectrogram(pos_aug)
            triplets.append((anchor_spec, pos_spec, neg_spec))

    if model.training != was_training:
        model.train() if was_training else model.eval()
    if was_training:
        model.train()

    avg_neg_dist = float(np.mean(neg_distances)) if neg_distances else 0.0
    print(f"  Epoch {epoch} — avg negative distance: {avg_neg_dist:.3f}  "
          f"({len(triplets)} mined triplets)")

    random.shuffle(triplets)
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

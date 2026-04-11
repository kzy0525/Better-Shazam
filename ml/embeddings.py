"""
ml/embeddings.py — FAISS index construction and querying.

Computes song-level embeddings using AudioEmbedder, stores them in a
FAISS IndexFlatIP (inner product = cosine similarity for L2-normalised
vectors), and provides query and update functions.
"""

import json
import os
from typing import List, Tuple

import faiss
import numpy as np
import soundfile as sf
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from audio.preprocess import full_preprocess_pipeline
from ml.dataset import slice_audio, _to_spectrogram, _spec_to_tensor
from config import SAMPLE_RATE, EMBEDDING_DIM


def _load_model(model_path: str):
    """Load AudioEmbedder weights, importing lazily to avoid circular imports."""
    from ml.model import get_model
    model = get_model(model_path)
    model.eval()
    return model


def _embed_audio(audio: np.ndarray, model, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Compute a single song-level embedding by averaging clip embeddings.

    Args:
        audio: 1-D float32 numpy array (mono, preprocessed).
        model: AudioEmbedder in eval() mode.
        sr:    Sample rate in Hz.

    Returns:
        1-D float32 numpy array of shape (EMBEDDING_DIM,), L2-normalised.
    """
    clips = slice_audio(audio, sr=sr, clip_length=5.0)
    if not clips:
        # Clip is shorter than 5s — use it as-is padded to 5s
        target = int(5.0 * sr)
        clips = [np.pad(audio, (0, max(0, target - len(audio))))[:target]]

    embeddings = []
    with torch.no_grad():
        for clip in clips:
            spec = _to_spectrogram(clip, sr)
            tensor = _spec_to_tensor(spec).unsqueeze(0)  # (1, 1, 128, T)
            emb = model(tensor).squeeze(0).numpy()       # (128,)
            embeddings.append(emb)

    avg = np.mean(embeddings, axis=0).astype(np.float32)
    # Re-normalise after averaging
    norm = np.linalg.norm(avg)
    if norm > 0:
        avg /= norm
    return avg


# ---------------------------------------------------------------------------
# Build index
# ---------------------------------------------------------------------------

def build_faiss_index(
    songs_dir: str = "songs/",
    model_path: str = "ml/embedder.pt",
    index_path: str = "ml/faiss.index",
    metadata_path: str = "ml/metadata.json",
) -> None:
    """Build a FAISS IndexFlatIP from all WAV files in songs_dir.

    Each song is represented by the average of its clip embeddings.
    Saves the index and a JSON metadata file mapping index positions
    to song title, artist, and filepath.

    Args:
        songs_dir:     Directory of WAV files (22050 Hz mono).
        model_path:    Path to saved AudioEmbedder weights.
        index_path:    Output path for the FAISS index.
        metadata_path: Output path for the JSON metadata file.
    """
    model = _load_model(model_path)

    wav_files = sorted([
        os.path.join(songs_dir, f)
        for f in os.listdir(songs_dir)
        if f.lower().endswith(".wav")
    ])

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    metadata = []

    for path in wav_files:
        stem = os.path.splitext(os.path.basename(path))[0]
        if " - " in stem:
            parts = stem.split(" - ", 1)
            artist, title = parts[0].strip(), parts[1].strip()
        else:
            title, artist = stem, "Unknown"

        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        audio = full_preprocess_pipeline(audio, sample_rate=sr)
        emb = _embed_audio(audio, model, sr=sr)

        index.add(emb.reshape(1, -1))
        metadata.append({"title": title, "artist": artist, "filepath": path})
        print(f"Indexed: {title} by {artist}")

    os.makedirs(os.path.dirname(index_path) if os.path.dirname(index_path) else ".", exist_ok=True)
    faiss.write_index(index, index_path)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nFAISS index saved: {index_path}  ({index.ntotal} songs)")
    print(f"Metadata saved:    {metadata_path}")


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def query_faiss(
    audio: np.ndarray,
    model_path: str = "ml/embedder.pt",
    index_path: str = "ml/faiss.index",
    metadata_path: str = "ml/metadata.json",
    top_k: int = 3,
) -> List[Tuple[str, str, float]]:
    """Identify a song snippet by searching the FAISS index.

    Args:
        audio:         1-D float32 numpy array (raw, will be preprocessed).
        model_path:    Path to saved AudioEmbedder weights.
        index_path:    Path to the FAISS index file.
        metadata_path: Path to the JSON metadata file.
        top_k:         Number of nearest neighbours to return.

    Returns:
        List of (title, artist, similarity_score) tuples, sorted by
        similarity descending.
    """
    model = _load_model(model_path)
    index = faiss.read_index(index_path)
    with open(metadata_path) as f:
        metadata = json.load(f)

    audio = full_preprocess_pipeline(audio, sample_rate=SAMPLE_RATE)
    emb = _embed_audio(audio, model, sr=SAMPLE_RATE).reshape(1, -1)

    k = min(top_k, index.ntotal)
    scores, indices = index.search(emb, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        meta = metadata[idx]
        results.append((meta["title"], meta["artist"], float(score)))

    return results


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------

def update_index(
    new_song_path: str,
    title: str,
    artist: str,
    model_path: str = "ml/embedder.pt",
    index_path: str = "ml/faiss.index",
    metadata_path: str = "ml/metadata.json",
) -> None:
    """Add a single new song to an existing FAISS index without rebuilding.

    Args:
        new_song_path: Path to a WAV file (22050 Hz mono).
        title:         Song title.
        artist:        Artist name.
        model_path:    Path to saved AudioEmbedder weights.
        index_path:    Path to the existing FAISS index.
        metadata_path: Path to the existing JSON metadata file.
    """
    model = _load_model(model_path)
    index = faiss.read_index(index_path)
    with open(metadata_path) as f:
        metadata = json.load(f)

    audio, sr = sf.read(new_song_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = full_preprocess_pipeline(audio, sample_rate=sr)
    emb = _embed_audio(audio, model, sr=sr).reshape(1, -1)

    index.add(emb)
    metadata.append({"title": title, "artist": artist, "filepath": new_song_path})

    faiss.write_index(index, index_path)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Added '{title}' by {artist} to index ({index.ntotal} songs total)")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_embeddings(
    songs_dir: str = "songs/",
    model_path: str = "ml/embedder.pt",
    save_path: str = "ml/embeddings_pca.png",
) -> None:
    """PCA scatter plot of per-clip embeddings coloured by song.

    Loads all WAV files, computes per-clip (not song-averaged) embeddings,
    reduces to 2D with PCA, and plots one cluster per song. Tight, well-
    separated clusters indicate the model learned meaningful representations.

    Args:
        songs_dir:  Directory of WAV files.
        model_path: Path to saved AudioEmbedder weights.
        save_path:  Path to save the PCA plot.
    """
    model = _load_model(model_path)

    wav_files = sorted([
        os.path.join(songs_dir, f)
        for f in os.listdir(songs_dir)
        if f.lower().endswith(".wav")
    ])

    all_embeddings = []
    all_labels = []

    for path in wav_files:
        stem = os.path.splitext(os.path.basename(path))[0]
        title = stem.split(" - ", 1)[1].strip() if " - " in stem else stem

        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = full_preprocess_pipeline(audio, sample_rate=sr)
        clips = slice_audio(audio, sr=sr, clip_length=5.0)

        with torch.no_grad():
            for clip in clips:
                spec = _to_spectrogram(clip, sr)
                tensor = _spec_to_tensor(spec).unsqueeze(0)
                emb = model(tensor).squeeze(0).numpy()
                all_embeddings.append(emb)
                all_labels.append(title)

    if len(all_embeddings) < 2:
        print("Not enough clips to visualise embeddings.")
        return

    X = np.stack(all_embeddings)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    unique_labels = list(dict.fromkeys(all_labels))
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))

    _, ax = plt.subplots(figsize=(10, 7))
    for i, label in enumerate(unique_labels):
        mask = [l == label for l in all_labels]
        pts = X_2d[mask]
        ax.scatter(pts[:, 0], pts[:, 1], s=40, color=cmap(i), label=label[:30], alpha=0.8)

    ax.set_title("Embedding Space — PCA (128D → 2D)", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"PCA embedding plot saved: {save_path}")
    plt.show()

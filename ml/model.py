"""
ml/model.py — AudioEmbedder CNN and triplet training loop.

Fine-tunes VGGish to produce 128-dim L2-normalised embeddings so that
clips of the same song (even at different tempos) map to nearby vectors
on the unit hypersphere.
"""

import os
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import audiomentations as A
import soundfile as sf

from ml.dataset import build_triplet_dataset, load_gtzan, slice_audio, _to_spectrogram, _spec_to_tensor
from config import SAMPLE_RATE


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AudioEmbedder(nn.Module):
    """VGGish backbone fine-tuned to produce 128-dim L2-normalised embeddings.

    Architecture:
        VGGish (pretrained, lower layers frozen) → linear(128) → L2 norm

    The forward pass accepts (batch, 1, 128, time_frames) mel spectrograms
    and returns (batch, 128) unit-norm embedding vectors.
    """

    def __init__(self):
        super().__init__()
        # Load VGGish from torch hub; suppress noisy download output
        vggish = torch.hub.load(
            "harritaylor/torchvggish", "vggish", pretrained=True, progress=False
        )

        # VGGish has .features (conv layers) and .embeddings (FC layers)
        self.features = vggish.features

        # Freeze all convolutional feature layers
        for param in self.features.parameters():
            param.requires_grad = False

        # Keep the first FC block from VGGish embeddings (4096 → 4096),
        # unfreeze only the last two FC layers
        self.fc1 = vggish.embeddings[0]   # Linear(12288, 4096)
        self.relu1 = vggish.embeddings[1]  # ReLU
        self.fc2 = vggish.embeddings[3]   # Linear(4096, 4096)  [2] is dropout
        self.relu2 = vggish.embeddings[4]  # ReLU

        # Freeze fc1, unfreeze fc2
        for param in self.fc1.parameters():
            param.requires_grad = False
        for param in self.fc2.parameters():
            param.requires_grad = True

        # New projection head: 4096 → 128
        self.projection = nn.Linear(4096, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute L2-normalised 128-dim embeddings.

        Args:
            x: Tensor of shape (batch, 1, 128, time_frames).

        Returns:
            Tensor of shape (batch, 128), unit-norm vectors.
        """
        # Repeat single channel to 3 channels — VGGish conv layers expect 1-ch
        # but were originally trained on log-mel which is single channel;
        # we keep it as-is since VGGish features expect (batch, 1, freq, time)
        out = self.features(x)           # (batch, C, H', W')
        out = out.flatten(start_dim=1)   # (batch, C*H'*W')
        out = self.relu1(self.fc1(out))  # (batch, 4096)
        out = self.relu2(self.fc2(out))  # (batch, 4096)
        out = self.projection(out)       # (batch, 128)
        out = F.normalize(out, p=2, dim=1)  # L2 normalise → unit hypersphere
        return out


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """Standard triplet loss with Euclidean distance.

    loss = mean(max(d(a, p) - d(a, n) + margin, 0))

    Args:
        anchor:   (batch, 128) embedding tensor.
        positive: (batch, 128) embedding tensor — same song as anchor.
        negative: (batch, 128) embedding tensor — different song.
        margin:   Minimum required separation between pos/neg distances.

    Returns:
        Scalar loss tensor.
    """
    d_pos = F.pairwise_distance(anchor, positive)
    d_neg = F.pairwise_distance(anchor, negative)
    loss = F.relu(d_pos - d_neg + margin)
    return loss.mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_model(model_path: str | None = None) -> AudioEmbedder:
    """Instantiate AudioEmbedder, optionally loading saved weights.

    Args:
        model_path: If provided, load weights from this .pt file.

    Returns:
        AudioEmbedder instance.
    """
    model = AudioEmbedder()
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Loaded model weights from: {model_path}")
    return model


def train(
    gtzan_dir: str = "data/genres_original/",
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    save_path: str = "ml/embedder.pt",
) -> AudioEmbedder:
    """Fine-tune AudioEmbedder using triplet loss on the GTZAN dataset.

    Loads the GTZAN dataset via load_gtzan(), builds a triplet dataset with
    cross-genre negatives, then trains with Adam and triplet loss.
    Prints per-epoch mean loss and fraction of triplets where the model
    correctly ordered anchor-positive closer than anchor-negative.
    Saves a checkpoint every 5 epochs and a training curves plot at the end.

    Args:
        gtzan_dir:  Path to the GTZAN genres_original/ folder.
        epochs:     Number of training epochs.
        batch_size: Batch size for DataLoader.
        lr:         Adam learning rate.
        save_path:  Path to save the final model weights.

    Returns:
        Trained AudioEmbedder in eval() mode.
    """
    entries = load_gtzan(gtzan_dir)
    dataset = build_triplet_dataset(entries)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = AudioEmbedder()
    model.train()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    history_loss: List[float] = []
    history_frac: List[float] = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for anchor, positive, negative in loader:
            optimizer.zero_grad()

            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            loss = triplet_loss(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(anchor)

            d_pos = F.pairwise_distance(emb_a, emb_p)
            d_neg = F.pairwise_distance(emb_a, emb_n)
            correct += (d_pos < d_neg).sum().item()
            total += len(anchor)

        mean_loss = epoch_loss / total
        frac_correct = correct / total
        history_loss.append(mean_loss)
        history_frac.append(frac_correct)

        print(f"Epoch {epoch:3d}/{epochs}  loss={mean_loss:.4f}  correct={frac_correct:.3f}")

        if epoch % 5 == 0:
            ckpt = save_path.replace(".pt", f"_epoch{epoch}.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  Checkpoint saved: {ckpt}")

    torch.save(model.state_dict(), save_path)
    print(f"Final model saved: {save_path}")

    # Training curves plot
    curves_path = os.path.join(os.path.dirname(save_path), "training_curves.png")
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(history_loss, color="steelblue", label="Triplet loss")
    ax2.plot(history_frac, color="darkorange", label="Fraction correct")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="steelblue")
    ax2.set_ylabel("Fraction correct", color="darkorange")
    ax1.set_title("Training Curves — AudioEmbedder", fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.tight_layout()
    plt.savefig(curves_path, dpi=150)
    print(f"Training curves saved: {curves_path}")
    plt.show()

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Tempo robustness evaluation
# ---------------------------------------------------------------------------

def evaluate_tempo_robustness(
    model: AudioEmbedder,
    songs_dir: str = "songs/",
    sr: int = SAMPLE_RATE,
    save_path: str = "ml/tempo_robustness.png",
) -> None:
    """Evaluate cosine similarity across tempo shifts for each song.

    For each song takes one clean clip as anchor, generates versions at
    0.80x, 0.90x, 1.00x, 1.10x, 1.20x speed, computes embeddings and
    prints similarity scores. Saves a bar chart.

    Args:
        model:     Trained AudioEmbedder in eval() mode.
        songs_dir: Directory of WAV files.
        sr:        Sample rate.
        save_path: Path to save the bar chart.
    """
    model.eval()
    rates = [0.80, 0.90, 1.00, 1.10, 1.20]
    wav_files = sorted([
        os.path.join(songs_dir, f)
        for f in os.listdir(songs_dir)
        if f.lower().endswith(".wav")
    ])

    all_similarities: dict[str, List[float]] = {}

    print(f"\n{'Song':<35} " + "  ".join(f"{r:.2f}x" for r in rates))
    print("-" * 70)

    for path in wav_files:
        name = os.path.splitext(os.path.basename(path))[0]
        audio, file_sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        clips = slice_audio(audio, sr=file_sr, clip_length=5.0)
        if not clips:
            continue
        anchor_clip = clips[len(clips) // 2]  # mid-song clip

        # Anchor embedding
        anchor_spec = _to_spectrogram(anchor_clip, sr)
        anchor_tensor = _spec_to_tensor(anchor_spec).unsqueeze(0)
        with torch.no_grad():
            anchor_emb = model(anchor_tensor)  # (1, 128)

        sims = []
        for rate in rates:
            stretcher = A.TimeStretch(min_rate=rate, max_rate=rate, p=1.0)
            stretched = stretcher(samples=anchor_clip, sample_rate=sr)
            # Trim/pad to original length
            if len(stretched) > len(anchor_clip):
                stretched = stretched[:len(anchor_clip)]
            else:
                stretched = np.pad(stretched, (0, len(anchor_clip) - len(stretched)))

            spec = _to_spectrogram(stretched.astype(np.float32), sr)
            tensor = _spec_to_tensor(spec).unsqueeze(0)
            with torch.no_grad():
                emb = model(tensor)
            sim = F.cosine_similarity(anchor_emb, emb).item()
            sims.append(sim)

        all_similarities[name] = sims
        row = "  ".join(f"{s:.3f}" for s in sims)
        print(f"{name[:34]:<35} {row}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(rates))
    width = 0.8 / max(len(all_similarities), 1)

    for i, (name, sims) in enumerate(all_similarities.items()):
        offset = (i - len(all_similarities) / 2) * width
        ax.bar(x + offset, sims, width, label=name[:25])

    ax.axhline(0.85, color="red", linestyle="--", linewidth=1, label="Target (0.85)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.2f}x" for r in rates])
    ax.set_xlabel("Tempo rate")
    ax.set_ylabel("Cosine similarity with anchor")
    ax.set_ylim(0, 1.05)
    ax.set_title("Tempo Robustness — Cosine Similarity vs. Tempo Shift", fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\nTempo robustness chart saved: {save_path}")
    plt.show()

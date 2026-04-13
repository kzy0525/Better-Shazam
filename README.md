# Better Shazam — Audio Fingerprinting System

A from-scratch audio identification system that replicates Shazam's core functionality, but better. 

Shazam identifies songs by converting audio into a spectrogram and matching frequency patterns against a database. This project replicates that pipeline from scratch, then extends it with a custom-trained neural network to handle noise and tempo-shifted audio that classical fingerprinting cannot identify.

---

## What It Does

Given a 10-second recording from your microphone, the system identifies which song is playing and returns the title, artist, and a confidence level.

It runs two identification paths simultaneously and combines their results:

- **Classical path** — extracts spectrogram peaks from the audio, hashes pairs of peaks into fingerprints, and looks them up in a SQLite database using a time-coherence voting algorithm. Fast and precise for clean audio.
- **ML path** — runs the audio through a trained CNN that maps mel spectrograms to 128-dimensional embedding vectors, then searches a FAISS index for the nearest match using cosine similarity. Robust to background noise and tempo shifts that break the classical path.

---

## How It Works

### Audio Pipeline
Raw microphone input is captured at 22050 Hz mono and passed through a two-stage noise removal pipeline before any fingerprinting occurs:

1. **Demucs source separation** — a pretrained neural network (htdemucs) separates the audio into stems (drums, bass, vocals, other) and recombines them. Anything that doesn't fit a musical stem — crowd noise, background talking, environmental sounds — is discarded.
2. **Wiener filter** — removes residual stationary noise (hiss, hum) left over after stem separation.
3. **Butterworth bandpass filter** (80–8000 Hz) — removes sub-bass rumble and high-frequency artifacts outside the musically relevant range.

### Classical Fingerprinting (Phase 3)
The cleaned audio is converted to a mel spectrogram (128 bins, STFT with n_fft=2048, hop=512). Local maxima are extracted as sparse peak points. Each anchor peak is paired with up to 15 nearby target peaks, and each pair is SHA-1 hashed into a 32-bit fingerprint encoding `(anchor_freq, target_freq, time_delta)`.

At query time, the snippet's hashes are looked up in SQLite and matched against the database using a time-offset coherence histogram — a genuine match produces many hashes that agree on the same playback position delta, while false positives scatter randomly.

### ML Embeddings (Phase 5)
A custom 4-layer CNN maps mel spectrograms to 128-dimensional L2-normalised embedding vectors. The model is trained with triplet loss on the GTZAN dataset (8 genres, 800 files): anchor clips are paired with augmented versions of the same clip as positives (including ±20% tempo shifts via TimeStretch) and clips from different genres as negatives.

Song embeddings are pre-computed and stored in a FAISS IndexFlatIP. At query time the snippet embedding is computed and the nearest neighbors are returned by cosine similarity.

**Key result:** the ML path correctly identifies songs at ±20% tempo shift with >0.93 cosine similarity — a case the classical path fundamentally cannot handle.

---

## Training Results

The model was trained on the GTZAN dataset (800 files across 8 genres). The 4 songs in `test_songs/` were not used during training — they serve as the held-out evaluation set to measure real-world identification performance.


| Epoch | Loss   | Fraction Correct |
|-------|--------|-----------------|
| 1     | 0.0039 | 0.998           |
| 2     | 0.0006 | 1.000           |
| 3     | 0.0003 | 1.000           |
| 4     | 0.0002 | 1.000           |
| 5     | 0.0002 | 1.000           |

### Tempo Robustness

The 4 songs in `test_songs/` were used to evaluate how well the model holds up when the same song plays at a different speed. Each song was run through the ML path at 5 tempo rates and compared against its own clean-speed embedding.

Cosine similarity between a clean anchor clip and tempo-shifted versions:

| Song | 0.80x | 0.90x | 1.00x | 1.10x | 1.20x |
|------|-------|-------|-------|-------|-------|
| Aespa - Whiplash | 0.930 | 0.973 | 1.000 | 0.974 | 0.953 |
| Katy Perry - California Gurls | 0.937 | 0.975 | 1.000 | 0.981 | 0.966 |
| Tchaikovsky - Piano Concerto 1 | 0.985 | 0.995 | 1.000 | 0.995 | 0.984 |
| The Weeknd - Save Your Tears | 0.970 | 0.988 | 1.000 | 0.985 | 0.968 |

All songs scored above 0.93 at ±20% tempo shift. Target threshold: 0.85.

---

## Project Structure

```
Better-Shazam/
├── audio/
│   ├── capture.py          # Microphone recording
│   └── preprocess.py       # Demucs + Wiener + bandpass pipeline
├── fingerprint/
│   ├── spectrogram.py      # Mel spectrogram + peak extraction
│   ├── hash.py             # Peak pairing and SHA-1 hashing
│   └── database.py         # SQLite fingerprint store
├── ml/
│   ├── dataset.py          # GTZAN loader + triplet dataset builder
│   ├── model.py            # CNN architecture + triplet training loop
│   └── embeddings.py       # FAISS index build, query, and visualisation
├── matcher.py              # Dual-path matcher (classical + ML, threaded)
├── main.py                 # Entry point — record and identify
├── config.py               # Shared constants
├── test_songs/             # Your local song library (MP3/WAV)
└── data/
    └── genres_original/    # GTZAN dataset (not committed)
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install pyroomacoustics
```

You also need ffmpeg for MP3 conversion:

```bash
brew install ffmpeg   # macOS
```

### 2. Download the GTZAN dataset (required for ML training only)

Download from Kaggle: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

Extract and place it at:

```
data/genres_original/
├── blues/
├── classical/
├── country/
...
```

---

## Usage

### Register your songs

Drop MP3/WAV files into `test_songs/` then register them:

```bash
python tests/test_phase3.py --songs test_songs/
```

This converts each file to 22050 Hz mono WAV, runs it through the full preprocessing pipeline, fingerprints it, and stores the hashes in `output/songs.db`. On subsequent runs use `--reuse-db` to skip re-registration.

### Identify a song

```bash
python main.py
```

Records 10 seconds from your microphone, runs both fingerprinting paths, and prints the result. Make sure the song is playing loudly enough to be picked up.

Optional flags:
```bash
python main.py --db output/songs.db --index ml/faiss.index
```

---

## Training the ML Model

Training only needs to be done once. The weights are saved to `ml/embedder.pt` and reused automatically on every subsequent run.

### Run training

```bash
python tests/test_phase5.py --skip-register
```

This will:
1. Load 800 files from GTZAN (8 genres)
2. Build ~7,200 triplets with augmentation (TimeStretch ±20%, noise, reverb, pitch shift)
3. Train the CNN for 5 epochs with triplet loss
4. Save weights to `ml/embedder.pt`
5. Build the FAISS index from your songs in `test_songs/`
6. Run the tempo robustness evaluation
7. Record a snippet and test the full dual-path matcher
8. Generate a PCA visualisation of the embedding space

If songs are already registered and you only want to retrain:
```bash
python tests/test_phase5.py --skip-register --epochs 5
```

If the model is already trained and you only want to rebuild the index and test matching:
```bash
python tests/test_phase5.py --skip-register --skip-train
```

---

## Key Constants

These are fixed across all modules and must stay consistent:

| Constant | Value |
|---|---|
| Sample rate | 22050 Hz |
| Clip length | 10 seconds |
| Channels | Mono |
| Mel bins | 128 |
| Embedding dimension | 128 |
| Hash fan-out | 15 |
| STFT n_fft | 2048 |
| STFT hop length | 512 |

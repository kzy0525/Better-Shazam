"""
Global configuration constants shared across all modules.
Do not change these values — they must remain consistent between
the fingerprinting database and any query made against it.
"""

# Audio
SAMPLE_RATE = 22050       # Hz
CLIP_LENGTH = 10          # seconds
CHANNELS = 1              # mono
FILE_FORMAT = "wav"

# Spectrogram
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 80                 # Hz — bandpass lower bound
FMAX = 8000               # Hz — bandpass upper bound

# Fingerprinting
PEAK_NEIGHBORHOOD = 20    # pixels — local maxima window size
FAN_VALUE = 15            # number of target peaks paired per anchor
MIN_HASH_TIME_DELTA = 0   # frames
MAX_HASH_TIME_DELTA = 200 # frames

# ML
EMBEDDING_DIM = 128

# Paths
SONGS_DIR = "songs"
DB_PATH = "fingerprints.db"
FAISS_INDEX_PATH = "embeddings.index"
FAISS_METADATA_PATH = "embeddings_meta.pkl"

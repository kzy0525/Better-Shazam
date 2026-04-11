from .model import get_model, train, triplet_loss, evaluate_tempo_robustness
from .embeddings import build_faiss_index, query_faiss, update_index, visualize_embeddings
from .dataset import build_triplet_dataset, slice_audio, augment_clip, verify_dataset

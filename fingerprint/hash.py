"""
fingerprint/hash.py — Peak pairing and hash generation.

Each pair of spectrogram peaks (anchor + target) within a time window
is encoded as a 32-bit hash of (anchor_freq, target_freq, time_delta).
The dense but consistent set of hashes is what gets stored in SQLite
and matched against at query time.
"""

import hashlib
import struct
from typing import List, Tuple


def generate_hashes(
    peaks: List[Tuple[int, int, float]],
    fan_out: int = 15,
    min_time_delta: int = 1,
    max_time_delta: int = 200,
) -> List[Tuple[int, int]]:
    """Generate fingerprint hashes from a list of spectrogram peaks.

    For each anchor peak, pairs it with up to fan_out target peaks whose
    time frames fall within [anchor_t + min_time_delta, anchor_t + max_time_delta].
    Each pair produces a 32-bit hash of (anchor_freq, target_freq, time_delta)
    and records the anchor's time frame as the offset.

    Args:
        peaks:          List of (time_frame, freq_bin, amplitude_db) tuples,
                        as returned by fingerprint.spectrogram.extract_peaks.
        fan_out:        Maximum number of target peaks to pair with each anchor.
        min_time_delta: Minimum time separation between anchor and target (frames).
        max_time_delta: Maximum time separation between anchor and target (frames).

    Returns:
        List of (hash_int, time_offset) tuples where hash_int is a 32-bit
        unsigned integer and time_offset is the anchor's time frame index.
    """
    # Sort by time so we can scan forward efficiently
    sorted_peaks = sorted(peaks, key=lambda p: p[0])
    n = len(sorted_peaks)
    hashes = []

    for i, (anchor_t, anchor_f, _) in enumerate(sorted_peaks):
        targets_found = 0

        for j in range(i + 1, n):
            target_t, target_f, _ = sorted_peaks[j]
            delta = target_t - anchor_t

            if delta < min_time_delta:
                continue
            if delta > max_time_delta:
                break  # list is sorted, no later peaks can qualify

            # Pack three 16-bit values into a bytes object and SHA-1 hash it
            raw = struct.pack(">HHH", anchor_f & 0xFFFF, target_f & 0xFFFF, delta & 0xFFFF)
            sha1 = hashlib.sha1(raw, usedforsecurity=False).digest()
            # Truncate to 32-bit unsigned integer (first 4 bytes, big-endian)
            hash_int = struct.unpack(">I", sha1[:4])[0]

            hashes.append((hash_int, anchor_t))
            targets_found += 1

            if targets_found >= fan_out:
                break

    return hashes


def hashes_to_hex(peaks: List[Tuple[int, int, float]]) -> List[Tuple[str, int]]:
    """Return hashes as hex strings instead of integers (for debugging only).

    Args:
        peaks: List of (time_frame, freq_bin, amplitude_db) tuples.

    Returns:
        List of (hex_string, time_offset) tuples.
    """
    return [(f"{h:08x}", t) for h, t in generate_hashes(peaks)]

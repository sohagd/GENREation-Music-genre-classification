
"""Shared audio preprocessing for training and inference."""

import numpy as np
import librosa

SAMPLE_RATE = 22050
DURATION_SECONDS = 30
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
MAX_FRAMES = int(np.ceil(
    SAMPLE_RATE * DURATION_SECONDS / HOP_LENGTH
))


def extract_mfcc(audio_path, sample_rate=SAMPLE_RATE, offset_seconds=0.0):
    """Extract a fixed-size MFCC feature map from an audio segment."""

    y, sr = librosa.load(
        str(audio_path),
        sr=sample_rate,
        mono=True,
        offset=max(0.0, float(offset_seconds)),
        duration=DURATION_SECONDS,
    )

    if y.size == 0:
        raise ValueError(
            "No audio found at the requested position."
        )

    target_samples = sample_rate * DURATION_SECONDS

    if len(y) < target_samples:
        y = np.pad(
            y,
            (0, target_samples - len(y))
        )
    else:
        y = y[:target_samples]

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    ).astype(np.float32)

    if mfcc.shape[1] < MAX_FRAMES:
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, MAX_FRAMES - mfcc.shape[1]))
        )
    else:
        mfcc = mfcc[:, :MAX_FRAMES]

    # Normalize each MFCC coefficient across time.
    mean = mfcc.mean(axis=1, keepdims=True)
    std = mfcc.std(axis=1, keepdims=True)

    mfcc = (mfcc - mean) / np.maximum(std, 1e-6)

    return mfcc[..., np.newaxis]
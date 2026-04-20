"""
P2(d): Compute average log-likelihood of sample-A (same speaker as training)
and sample-B (different speaker) under the K=32 GMM fitted in P2(b).

sample-A: audio/R1.wav from Assignment 2 (same speaker, new recording).
sample-B: place your friend's 8-kHz recording at P2/audio/sample_B.wav
          (any format readable by soundfile; will be resampled to 8 kHz).

If sample_B.wav is absent, the script exits with an informative message.
"""

import os, sys
import numpy as np
import librosa
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmm import GMM

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P2_DIR    = os.path.join(BASE_DIR, "P2")
A2_AUDIO  = os.path.join(BASE_DIR, "audio")

SAMPLE_A_PATH = os.path.join(A2_AUDIO,     "R1.wav")
SAMPLE_B_PATH = os.path.join(P2_DIR, "audio", "sample_B.wav")

TARGET_SR = 8000
N_MFCC    = 13
N_FFT     = 256
HOP       = 128
N_MELS    = 26


# ── Helpers ────────────────────────────────────────────────────────────────

def load_mfcc(path: str, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Load audio → mono 8 kHz → 13-dim MFCC (normalised)."""
    audio, sr = sf.read(path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    mfcc = librosa.feature.mfcc(
        y=audio, sr=TARGET_SR,
        n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
    ).T                                      # (T, 13)
    return (mfcc - mu) / std


# ── Load model ─────────────────────────────────────────────────────────────
print("Loading K=32 GMM ...")
ckpt  = np.load(os.path.join(P2_DIR, "gmm32_model.npz"))
gmm   = GMM(n_components=32)
gmm.weights_ = ckpt["weights"]
gmm.means_   = ckpt["means"]
gmm.covs_    = ckpt["covs"]
mu_X  = ckpt["mu_X"]
std_X = ckpt["std_X"]

# ── Check sample-B exists ──────────────────────────────────────────────────
if not os.path.exists(SAMPLE_B_PATH):
    print(f"\n[!] sample-B not found at: {SAMPLE_B_PATH}")
    print("    Place your friend's speech recording there and re-run.")
    print("    Supported formats: WAV, FLAC, OGG (anything soundfile reads).")
    sys.exit(0)

# ── Compute average log-likelihoods ───────────────────────────────────────
print("\nComputing average log-likelihoods ...")

for label, path in [("sample-A (same speaker)", SAMPLE_A_PATH),
                    ("sample-B (diff. speaker)", SAMPLE_B_PATH)]:
    X = load_mfcc(path, mu_X, std_X)
    total_ll   = gmm.score(X)
    avg_ll     = total_ll / len(X)
    dur        = len(X) * HOP / TARGET_SR
    print(f"\n  {label}")
    print(f"    Path    : {path}")
    print(f"    Frames  : {len(X)}  (~{dur:.1f} s)")
    print(f"    Total LL: {total_ll:.2f}")
    print(f"    Avg LL  : {avg_ll:.4f}  (per frame)")

print("\nDone.")

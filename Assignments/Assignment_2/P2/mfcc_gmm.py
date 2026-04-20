"""
P2(b): Extract 13-dim MFCC features from 5 Assignment-1 speech samples,
fit GMM for K in {1,2,4,8,16,32,64}, plot total log-likelihood vs K.
"""

import os, sys
import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# add P2 dir to path so we can import our GMM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmm import GMM

# ── Config ─────────────────────────────────────────────────────────────────
A1_AUDIO = "/home/anupam/Desktop/IIT_LECTURES/Year_4/Sem_8/EE_679/Assignments/Assignment_1/audio_sample"
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
IMG_DIR  = os.path.join(OUT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

TARGET_SR = 8000
N_MFCC    = 13
N_FFT     = 256          # 32 ms frame at 8 kHz
HOP       = 128          # 50 % overlap
N_MELS    = 26

SPEECH_SAMPLES = [
    "sample_1_laptop_microphone.wav",
    "sample_2_WH720N_headphones_with_ANC.wav",
    "sample_2_WH720N_headphones_without_ANC.wav",
    "sample_3_K8_wireless_microphone.wav",
    "sample_3_K8_wireless_microphone_take_2.wav",
]

K_VALUES = [1, 2, 4, 8, 16, 32, 64]
N_RUNS   = 3          # restarts per K to avoid local optima
RANDOM_SEED = 42


# ── Step 1: Load and resample ──────────────────────────────────────────────
print("=" * 60)
print("Step 1: Loading and resampling to 8 kHz")
print("=" * 60)

all_mfcc = []
for fname in SPEECH_SAMPLES:
    path = os.path.join(A1_AUDIO, fname)
    audio, sr = sf.read(path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)          # stereo → mono
    audio = audio.astype(np.float32)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    dur = len(audio) / TARGET_SR
    print(f"  {fname}  →  {dur:.1f}s @ {TARGET_SR} Hz")
    all_mfcc.append(audio)

# ── Step 2: Extract MFCCs ──────────────────────────────────────────────────
print("\nStep 2: Extracting 13-dim MFCCs")

feature_list = []
for i, (audio, fname) in enumerate(zip(all_mfcc, SPEECH_SAMPLES)):
    mfcc = librosa.feature.mfcc(
        y=audio, sr=TARGET_SR,
        n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
    )                                       # (13, T)
    mfcc = mfcc.T                           # (T, 13)
    feature_list.append(mfcc)
    print(f"  {fname}:  {mfcc.shape[0]} frames × {mfcc.shape[1]} dims")

X = np.vstack(feature_list)                # (N_total, 13)
print(f"\n  Total feature matrix: {X.shape}")

# Standardise (zero mean, unit variance per dimension)
mu_X  = X.mean(axis=0)
std_X = X.std(axis=0) + 1e-8
X_norm = (X - mu_X) / std_X

# ── Step 3: Fit GMM for each K ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: Fitting GMM for K = " + str(K_VALUES))
print("=" * 60)

log_likelihoods = []
best_models     = {}

for K in K_VALUES:
    best_ll    = -np.inf
    best_model = None
    for run in range(N_RUNS):
        g = GMM(n_components=K, max_iter=200, tol=1e-4,
                reg_covar=1e-4, random_state=RANDOM_SEED + run)
        g.fit(X_norm)
        if g.log_likelihood_ > best_ll:
            best_ll    = g.log_likelihood_
            best_model = g
    log_likelihoods.append(best_ll)
    best_models[K] = best_model
    print(f"  K={K:3d}  log-likelihood = {best_ll:14.2f}  "
          f"(converged in {best_model.n_iter_} iters)")

# ── Step 4: Plot ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(K_VALUES, log_likelihoods, "o-", color="#2563EB", linewidth=2,
        markersize=8, markerfacecolor="white", markeredgewidth=2)
for K, ll in zip(K_VALUES, log_likelihoods):
    ax.annotate(f"{ll:.0f}", (K, ll),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=8)
ax.set_xscale("log", base=2)
ax.set_xticks(K_VALUES)
ax.set_xticklabels([str(k) for k in K_VALUES])
ax.set_xlabel("Number of GMM components (K)", fontsize=12)
ax.set_ylabel("Total log-likelihood", fontsize=12)
ax.set_title("GMM log-likelihood vs. number of mixtures\n"
             "(13-dim MFCC, 5 speech samples, 8 kHz)", fontsize=13)
ax.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plot_path = os.path.join(IMG_DIR, "p2b_gmm_loglikelihood.png")
fig.savefig(plot_path, dpi=150)
print(f"\nPlot saved → {plot_path}")

# ── Step 5: Print final table ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("Results Table")
print("=" * 60)
print(f"{'K':>4}  {'Log-likelihood':>18}  {'Delta LL':>12}")
print("-" * 40)
prev = None
for K, ll in zip(K_VALUES, log_likelihoods):
    delta = f"{ll - prev:+.2f}" if prev is not None else "—"
    print(f"{K:>4}  {ll:>18.2f}  {delta:>12}")
    prev = ll

# Save raw numbers for README
np.save(os.path.join(OUT_DIR, "p2b_ll_results.npy"),
        np.array(list(zip(K_VALUES, log_likelihoods))))

# ── Step 6: Save K=32 model and feature normalisation stats ───────────────
gmm32 = best_models[32]
model_path = os.path.join(OUT_DIR, "gmm32_model.npz")
np.savez(model_path,
         weights=gmm32.weights_,
         means=gmm32.means_,
         covs=gmm32.covs_,
         mu_X=mu_X,
         std_X=std_X)
print(f"K=32 model saved → {model_path}")
print("\nDone.")

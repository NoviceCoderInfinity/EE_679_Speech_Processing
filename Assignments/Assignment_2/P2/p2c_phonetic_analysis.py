"""
P2(c): Inspect the 32-mixture GMM components and check whether they
correspond to different phonetic classes.

Approach:
- Load the saved K=32 GMM (fitted in P2b).
- Parse the IPA phoneme-tier TextGrid for sample_2 (the only annotated sample).
- Extract 13-dim MFCCs from sample_2 at 8 kHz and assign each frame a
  phoneme label based on frame timestamp.
- Compute responsibilities and hard-assign each frame to the dominant
  component.
- Aggregate: for each component, tally which phonetic classes dominate.
- Visualise component means (MFCC heatmap) and a component-vs-class heatmap.
"""

import os, sys, re
import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmm import GMM

# ── Paths ──────────────────────────────────────────────────────────────────
A1_AUDIO = "/home/anupam/Desktop/IIT_LECTURES/Year_4/Sem_8/EE_679/Assignments/Assignment_1/audio_sample"
P2_DIR   = os.path.dirname(os.path.abspath(__file__))
IMG_DIR  = os.path.join(P2_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

TARGET_SR = 8000
N_MFCC    = 13
N_FFT     = 256
HOP       = 128
N_MELS    = 26

TEXTGRID_FILE = os.path.join(A1_AUDIO, "sample_2_WH720N_headphones_with_ANC.TextGrid")
AUDIO_FILE    = os.path.join(A1_AUDIO, "sample_2_WH720N_headphones_with_ANC.wav")


# ── Step 1: Load saved GMM ─────────────────────────────────────────────────
print("Loading K=32 GMM ...")
ckpt = np.load(os.path.join(P2_DIR, "gmm32_model.npz"))
gmm  = GMM(n_components=32)
gmm.weights_ = ckpt["weights"]
gmm.means_   = ckpt["means"]
gmm.covs_    = ckpt["covs"]
mu_X  = ckpt["mu_X"]
std_X = ckpt["std_X"]
print(f"  weights: {gmm.weights_.shape}  means: {gmm.means_.shape}  covs: {gmm.covs_.shape}")


# ── Step 2: Parse TextGrid phoneme tier ────────────────────────────────────

def parse_textgrid_phoneme_tier(path: str):
    """Returns list of (xmin, xmax, label) tuples from the 'phenome' tier."""
    with open(path, encoding="utf-16") as f:
        content = f.read()
    lines = [l.strip() for l in content.splitlines()]

    # Find start of phenome tier
    tier_start = None
    for i, l in enumerate(lines):
        if 'name = "phenome"' in l:
            tier_start = i
            break
    if tier_start is None:
        raise ValueError("No 'phenome' tier found in TextGrid")

    intervals = []
    i = tier_start
    while i < len(lines):
        if re.match(r"intervals \[\d+\]:", lines[i]):
            xmin = float(lines[i+1].split("=")[1].strip())
            xmax = float(lines[i+2].split("=")[1].strip())
            text = lines[i+3].split("=")[1].strip().strip('"')
            intervals.append((xmin, xmax, text))
            i += 4
        else:
            i += 1
    return intervals


print("Parsing TextGrid ...")
phoneme_intervals = parse_textgrid_phoneme_tier(TEXTGRID_FILE)
print(f"  {len(phoneme_intervals)} phoneme intervals")

# Unique non-silence labels
unique_labels = sorted(set(t for _, _, t in phoneme_intervals if t))
print(f"  Unique phoneme symbols: {unique_labels}")


# ── Step 3: Broad phonetic class mapping ───────────────────────────────────

def classify_phoneme(sym: str) -> str:
    """Map IPA symbol → broad phonetic class."""
    if not sym:
        return "silence"
    # Vowels (IPA + common combinations)
    vowel_chars = set("ɑɐɛeəɜɪiɔoʊuæʌɵaɐ")
    if any(c in vowel_chars for c in sym):
        return "vowel"
    # Fricatives
    if any(c in sym for c in ["ʄ", "ˢ", "s", "z", "ʃ", "ʒ", "f", "v", "θ", "ð", "h"]):
        return "fricative"
    # Stops
    if any(c in sym for c in ["p", "b", "ʈ", "t", "d", "k", "g", "ʔ"]):
        return "stop"
    # Nasals
    if any(c in sym for c in ["m", "n", "ŋ", "ɲ"]):
        return "nasal"
    # Approximants / liquids
    if any(c in sym for c in ["ɹ", "l", "r", "w", "j"]):
        return "approximant"
    return "other"


# Build label→class lookup
label_to_class = {lbl: classify_phoneme(lbl) for lbl in unique_labels}
label_to_class[""] = "silence"
phonetic_classes = ["silence", "vowel", "fricative", "stop", "nasal", "approximant", "other"]
class_colors     = ["#9CA3AF", "#3B82F6", "#F59E0B", "#EF4444", "#10B981", "#8B5CF6", "#EC4899"]

print("\nPhoneme → class mapping:")
for lbl, cls in sorted(label_to_class.items()):
    if lbl:
        print(f"  {lbl!r:12s} → {cls}")


# ── Step 4: Load audio and extract MFCCs with timestamps ──────────────────
audio, sr = sf.read(AUDIO_FILE)
if audio.ndim == 2:
    audio = audio.mean(axis=1)
audio = audio.astype(np.float32)
if sr != TARGET_SR:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

mfcc = librosa.feature.mfcc(
    y=audio, sr=TARGET_SR,
    n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
).T                                          # (T, 13)

# Centre timestamps for each frame
T = mfcc.shape[0]
frame_times = librosa.frames_to_time(np.arange(T), sr=TARGET_SR, hop_length=HOP, n_fft=N_FFT)

# Assign phoneme label to each frame
def get_phoneme_at(t, intervals):
    for xmin, xmax, label in intervals:
        if xmin <= t < xmax:
            return label
    return ""

frame_labels  = [get_phoneme_at(t, phoneme_intervals) for t in frame_times]
frame_classes = [label_to_class.get(lbl, "other") for lbl in frame_labels]

# Normalise and get responsibilities
X_norm = (mfcc - mu_X) / std_X
resp   = gmm.predict_proba(X_norm)          # (T, 32)
assignments = resp.argmax(axis=1)           # hard assignment


# ── Step 5: Component vs phonetic class heatmap ───────────────────────────
K = 32
class_counts = np.zeros((K, len(phonetic_classes)))
for comp, cls in zip(assignments, frame_classes):
    ci = phonetic_classes.index(cls)
    class_counts[comp, ci] += 1

# Normalise rows so each component sums to 1
row_sums = class_counts.sum(axis=1, keepdims=True)
class_prob = np.divide(class_counts, row_sums, where=row_sums > 0)

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(class_prob, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
ax.set_xticks(range(len(phonetic_classes)))
ax.set_xticklabels(phonetic_classes, rotation=30, ha="right", fontsize=10)
ax.set_yticks(range(K))
ax.set_yticklabels([f"C{k}" for k in range(K)], fontsize=8)
ax.set_xlabel("Phonetic class", fontsize=12)
ax.set_ylabel("GMM component", fontsize=12)
ax.set_title("Component vs phonetic class (fraction of frames)\nK=32 GMM, sample_2", fontsize=12)
plt.colorbar(im, ax=ax, label="Fraction of frames")
plt.tight_layout()
heatmap_path = os.path.join(IMG_DIR, "p2c_component_phoneme_heatmap.png")
fig.savefig(heatmap_path, dpi=150)
print(f"\nHeatmap saved → {heatmap_path}")


# ── Step 6: Component mean MFCC heatmap ───────────────────────────────────
# Denormalise means back to MFCC scale for interpretability
means_mfcc = gmm.means_ * std_X + mu_X     # (32, 13)

# Sort components by dominant class for visual coherence
dominant_class_idx = class_prob.argmax(axis=1)
sort_order = np.argsort(dominant_class_idx)

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(means_mfcc[sort_order], aspect="auto", cmap="RdBu_r")
ax.set_xlabel("MFCC coefficient index (0=C0 … 12=C12)", fontsize=11)
ax.set_ylabel("GMM component (sorted by dominant class)", fontsize=11)
ax.set_title("Mean MFCC vectors of K=32 components\n(sorted by dominant phonetic class)", fontsize=12)
ax.set_xticks(range(N_MFCC))
ax.set_xticklabels([f"C{i}" for i in range(N_MFCC)], fontsize=8)
ax.set_yticks(range(K))
sorted_labels = [f"C{i}({phonetic_classes[dominant_class_idx[i]]})"
                 for i in sort_order]
ax.set_yticklabels(sorted_labels, fontsize=7)
plt.colorbar(im, ax=ax, label="MFCC value (de-normalised)")
plt.tight_layout()
means_path = os.path.join(IMG_DIR, "p2c_component_means.png")
fig.savefig(means_path, dpi=150)
print(f"Means heatmap saved → {means_path}")


# ── Step 7: Summary statistics ────────────────────────────────────────────
print("\n" + "=" * 65)
print("Component dominant-class summary")
print("=" * 65)
print(f"{'Comp':>5}  {'Dominant class':>14}  {'Frac':>6}  {'#frames':>8}  "
      f"{'Top phonemes'}")
print("-" * 65)
for k in range(K):
    dc_idx = class_prob[k].argmax()
    dc     = phonetic_classes[dc_idx]
    frac   = class_prob[k, dc_idx]
    n_fr   = int(class_counts[k].sum())
    # top-3 phoneme symbols in this component
    sym_counts = {}
    for sym, comp in zip(frame_labels, assignments):
        if comp == k and sym:
            sym_counts[sym] = sym_counts.get(sym, 0) + 1
    top_syms = sorted(sym_counts, key=sym_counts.get, reverse=True)[:4]
    print(f"  C{k:2d}  {dc:>14s}  {frac:>5.1%}  {n_fr:>8d}  {' '.join(top_syms)}")

# Class coverage: how many components cover each class
print("\nClass coverage (components where this class is dominant):")
for ci, cls in enumerate(phonetic_classes):
    comps = [k for k in range(K) if class_prob[k].argmax() == ci]
    print(f"  {cls:>12s}: {len(comps):2d} components  {comps}")

print("\nDone.")

"""
P1(b): Neural network-based speech enhancement using DeepFilterNet.

Enhances R1, R2, R3 with DeepFilterNet, saves outputs to P1/neural/,
then computes SNR, PESQ, and STOI metrics comparing classical vs neural.

Reference for metrics: original R1 (least-noisy recording) downmixed to mono
at 16 kHz, which serves as the pseudo-clean reference.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import librosa
from pesq import pesq
from pystoi import stoi

from df.enhance import enhance, init_df, load_audio, save_audio
from df.io import resample

# ── Paths ──────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO   = os.path.join(BASE, "audio")
CLS_DIR = os.path.join(BASE, "P1", "classical")
NEU_DIR = os.path.join(BASE, "P1", "neural")
os.makedirs(NEU_DIR, exist_ok=True)

RECORDINGS = ["R1", "R2", "R3"]
METRIC_SR  = 16000   # PESQ wideband + STOI reference rate


# ── Helpers ────────────────────────────────────────────────────────────────

def load_mono_16k(path: str) -> np.ndarray:
    """Load audio, downmix to mono, resample to METRIC_SR."""
    audio, sr = sf.read(path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != METRIC_SR:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=METRIC_SR)
    return audio.astype(np.float64)


def compute_snr(ref: np.ndarray, deg: np.ndarray) -> float:
    """
    Signal-to-Noise Ratio: treat `ref` as signal, (deg - ref) as noise.
    Aligns lengths before computation.
    """
    n = min(len(ref), len(deg))
    r, d = ref[:n], deg[:n]
    noise = d - r
    signal_power = np.mean(r ** 2)
    noise_power  = np.mean(noise ** 2)
    if noise_power < 1e-12:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)


def compute_pesq(ref: np.ndarray, deg: np.ndarray) -> float:
    n = min(len(ref), len(deg))
    try:
        return pesq(METRIC_SR, ref[:n], deg[:n], "wb")
    except Exception as e:
        print(f"    PESQ failed: {e}")
        return float("nan")


def compute_stoi(ref: np.ndarray, deg: np.ndarray) -> float:
    n = min(len(ref), len(deg))
    try:
        return stoi(ref[:n], deg[:n], METRIC_SR, extended=False)
    except Exception as e:
        print(f"    STOI failed: {e}")
        return float("nan")


# ── Step 1: DeepFilterNet enhancement ─────────────────────────────────────

print("=" * 60)
print("Step 1: Enhancing with DeepFilterNet")
print("=" * 60)

model, df_state, _ = init_df()
df_sr = df_state.sr()   # 48000 Hz
print(f"DeepFilterNet sample rate: {df_sr} Hz\n")

for rec in RECORDINGS:
    src  = os.path.join(AUDIO, f"{rec}.wav")
    dst  = os.path.join(NEU_DIR, f"{rec}.wav")
    print(f"  Enhancing {rec} ...")

    # load_audio resamples to df_sr automatically
    audio, _ = load_audio(src, sr=df_sr)
    enhanced  = enhance(model, df_state, audio)
    save_audio(dst, enhanced, df_sr)
    print(f"    Saved -> {dst}")

print()


# ── Step 2: Metrics ────────────────────────────────────────────────────────
# Two comparison strategies:
#  (A) All recordings vs original R1 as pseudo-clean reference — valid for
#      R1 only (same speech); R2/R3 have different content so PESQ/STOI are
#      not meaningful there, but SNR still indicates spectral similarity.
#  (B) Within each recording: classical enhanced vs neural enhanced, using
#      the classical output as reference — shows relative difference between
#      the two enhancement methods on the same speech.

print("=" * 60)
print("Step 2: Computing metrics")
print("=" * 60)

results_A = {}  # vs R1 reference
results_B = {}  # classical vs neural per recording

ref_r1 = load_mono_16k(os.path.join(AUDIO, "R1.wav"))

# ── (A) vs R1 reference ────────────────────────────────────────────────────
print("\n[A] Reference = original R1  (meaningful for R1; spectral proxy for R2/R3)")
for rec in RECORDINGS:
    results_A[rec] = {}
    print(f"\n  {rec}:")
    orig  = load_mono_16k(os.path.join(AUDIO, f"{rec}.wav"))
    cls_  = load_mono_16k(os.path.join(CLS_DIR, f"{rec}.wav"))
    neu_  = load_mono_16k(os.path.join(NEU_DIR, f"{rec}.wav"))

    for label, sig in [("Original", orig), ("Classical", cls_), ("Neural", neu_)]:
        snr_val  = compute_snr(ref_r1, sig)
        pesq_val = compute_pesq(ref_r1, sig)
        stoi_val = compute_stoi(ref_r1, sig)
        results_A[rec][label] = {"SNR": snr_val, "PESQ": pesq_val, "STOI": stoi_val}
        print(f"    [{label:9s}]  SNR={snr_val:+7.2f} dB  PESQ={pesq_val:.3f}  STOI={stoi_val:.4f}")

# ── (B) classical vs neural within each recording ─────────────────────────
print("\n[B] Reference = classical enhanced  (direct method comparison)")
for rec in RECORDINGS:
    cls_  = load_mono_16k(os.path.join(CLS_DIR, f"{rec}.wav"))
    neu_  = load_mono_16k(os.path.join(NEU_DIR, f"{rec}.wav"))

    snr_val  = compute_snr(cls_, neu_)
    pesq_val = compute_pesq(cls_, neu_)
    stoi_val = compute_stoi(cls_, neu_)
    results_B[rec] = {"SNR": snr_val, "PESQ": pesq_val, "STOI": stoi_val}
    print(f"  {rec}  Neural vs Classical:  SNR={snr_val:+7.2f} dB  PESQ={pesq_val:.3f}  STOI={stoi_val:.4f}")

print()
print("=" * 60)
print("Summary Table (A): all methods vs original R1")
print("=" * 60)
header = f"{'Recording':<10} {'Method':<12} {'SNR (dB)':>10} {'PESQ':>8} {'STOI':>8}"
print(header)
print("-" * len(header))
for rec in RECORDINGS:
    for method in ["Original", "Classical", "Neural"]:
        m = results_A[rec][method]
        print(f"{rec:<10} {method:<12} {m['SNR']:>+10.2f} {m['PESQ']:>8.3f} {m['STOI']:>8.4f}")
    print()

print("=" * 60)
print("Summary Table (B): neural vs classical (per recording)")
print("=" * 60)
print(f"{'Recording':<10} {'SNR (dB)':>10} {'PESQ':>8} {'STOI':>8}")
print("-" * 40)
for rec in RECORDINGS:
    m = results_B[rec]
    print(f"{rec:<10} {m['SNR']:>+10.2f} {m['PESQ']:>8.3f} {m['STOI']:>8.4f}")

# ── Step 3: Save metrics to CSV ────────────────────────────────────────────
import csv
csv_path = os.path.join(BASE, "P1", "metrics.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Recording", "Method", "Ref", "SNR_dB", "PESQ", "STOI"])
    for rec in RECORDINGS:
        for method in ["Original", "Classical", "Neural"]:
            m = results_A[rec][method]
            writer.writerow([rec, method, "R1_original",
                             f"{m['SNR']:.4f}", f"{m['PESQ']:.4f}", f"{m['STOI']:.4f}"])
    for rec in RECORDINGS:
        m = results_B[rec]
        writer.writerow([rec, "Neural", "Classical_enhanced",
                         f"{m['SNR']:.4f}", f"{m['PESQ']:.4f}", f"{m['STOI']:.4f}"])
print(f"\nMetrics saved -> {csv_path}")

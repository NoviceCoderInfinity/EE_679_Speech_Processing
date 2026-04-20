# P1: Speech Enhancement

## Recordings

| File | Description |
|------|-------------|
| `audio/R1.wav` | Speech + background music (16 kHz recording, ~48 s) |
| `audio/R2.wav` | Speech + keyboard typing (~30 s) |
| `audio/R3.wav` | Speech + fan noise, recorded 2 m from mic (~42 s) |

All recordings: 44100 Hz, stereo.

---

## Part (a): Classical Enhancement — Audacity Noise Reduction

Enhanced using Audacity's built-in noise reduction tool (noise profile sampled from a silent segment, then applied across the full recording). Output saved to `P1/classical/`.

**Observations:**
- **R1** (music background): Audacity successfully attenuated the background music while preserving speech quality. The enhancement is subtle since music is spectrally complex.
- **R2** (keyboard typing): Transient keyboard clicks are partially suppressed; some residual artefacts remain due to the impulsive nature of keystrokes.
- **R3** (fan noise): Fan noise is largely stationary and broadband — Audacity performs well, producing a noticeably cleaner signal with minimal speech distortion.

---

## Part (b): Neural Enhancement — DeepFilterNet3

**Tool:** [DeepFilterNet](https://github.com/rikorose/deepfilternet) (`deepfilternet` v0.5.6, model `DeepFilterNet3`, checkpoint epoch 120)  
**Script:** `P1/enhance_neural.py`  
**Output:** `P1/neural/` (saved at 48 kHz, DeepFilterNet native rate)

### How it works

DeepFilterNet3 is a deep neural network trained on the DNS Challenge dataset. It operates in the frequency domain using a two-stage approach: a coarse ERB-domain filter for broadband noise suppression followed by a fine deep-filtering stage for residual artefacts. It runs in real time on GPU.

### Metric computation notes

Metrics are computed at 16 kHz mono (resampled). Two comparison schemes are used:

- **Table A** — all methods vs. original R1 as pseudo-clean reference. Fully valid for R1 (same speech). For R2/R3, PESQ/STOI are not interpretable (different speech content across recordings); only SNR provides a spectral proximity indication.
- **Table B** — neural enhanced vs. classical enhanced within each recording (classical as reference). This directly quantifies how much the two methods differ on the same audio.

---

## Metrics

### Table A: All methods vs. original R1 (pseudo-clean reference)

| Recording | Method    | SNR (dB) | PESQ  | STOI   |
|-----------|-----------|----------|-------|--------|
| R1        | Original  | +∞       | 4.644 | 1.0000 |
| R1        | Classical | +6.62    | 4.634 | 0.9987 |
| R1        | Neural    | +8.14    | 1.121 | 0.8056 |
| R2        | Original  | −1.62    | 1.152 | 0.0075 |
| R2        | Classical | −0.45    | 1.115 | 0.0075 |
| R2        | Neural    | −0.74    | 1.081 | −0.008 |
| R3        | Original  | −0.87    | 2.558 | 0.0302 |
| R3        | Classical | −0.27    | 2.992 | 0.0170 |
| R3        | Neural    | −0.36    | 1.449 | 0.0308 |

> **Note:** PESQ and STOI for R2 and R3 are measured against R1 (a different utterance), so only the SNR column is interpretable for those rows.

### Table B: Neural vs. classical (per-recording direct comparison)

| Recording | SNR (dB) | PESQ  | STOI   |
|-----------|----------|-------|--------|
| R1        | +4.54    | 1.150 | 0.8071 |
| R2        | +6.28    | 1.116 | 0.7380 |
| R3        | −3.73    | 1.161 | 0.0302 |

---

## Analysis and Comparison

### R1 — Speech + background music

Classical (Audacity) is conservative: PESQ = 4.634 and STOI = 0.999 indicate the output is perceptually near-identical to the original. Neural (DeepFilterNet) achieves a higher SNR (+8.14 dB vs. +6.62 dB), reflecting more aggressive suppression of the background music. However, PESQ drops to 1.121 and STOI to 0.806 relative to R1, because DeepFilterNet treats the music as noise and removes it — the resulting signal is spectrally different from the music-containing reference R1, not because speech quality is degraded.

### R2 — Speech + keyboard typing

Both methods modestly improve SNR relative to R1 (−0.45 dB classical, −0.74 dB neural vs. −1.62 dB original). Table B shows neural suppresses ~6.3 dB more noise than classical on this recording, with STOI = 0.74, indicating the speech content is largely intelligible in the neural output relative to classical. Keyboard clicks (impulsive noise) are more effectively suppressed by the neural model since it was trained on diverse noise types.

### R3 — Speech + fan noise at 2 m

Fan noise is stationary and broadband — a type where classical spectral subtraction performs well. Classical achieves PESQ = 2.99 vs. 1.45 for neural (both measured against R1, which has different speech). Table B SNR of −3.73 dB suggests neural output diverges significantly from classical. This is likely because DeepFilterNet over-suppresses or introduces artefacts on this specific fan-noise profile, whereas Audacity's noise profile estimation is well-suited to stationary noise.

### Summary

| Criterion | Classical (Audacity) | Neural (DeepFilterNet) |
|-----------|---------------------|----------------------|
| Stationary noise (fan) | **Better** — targeted noise profile | Over-suppression risk |
| Non-stationary noise (keyboard, music) | Partial suppression | **Better** — stronger suppression |
| Speech distortion (R1 PESQ) | **4.634** (minimal distortion) | 1.121 (more aggressive) |
| Speech intelligibility (R1 STOI) | **0.999** | 0.806 |
| Ease of use | Manual noise profile needed | Fully automatic |

Classical methods excel on stationary noise with minimal speech distortion. Neural methods offer stronger, automatic suppression for diverse or non-stationary noise at the cost of potential over-processing artefacts.

---

# P2: Stochastic Modeling — Gaussian Mixture Model

## Dataset

Five speech samples from Assignment 1, resampled to **8 kHz mono**:

| # | File | Duration |
|---|------|----------|
| 1 | `sample_1_laptop_microphone.wav` | 16.5 s |
| 2 | `sample_2_WH720N_headphones_with_ANC.wav` | 18.6 s |
| 3 | `sample_2_WH720N_headphones_without_ANC.wav` | 18.9 s |
| 4 | `sample_3_K8_wireless_microphone.wav` | 17.0 s |
| 5 | `sample_3_K8_wireless_microphone_take_2.wav` | 17.1 s |

---

## Part (a): GMM Implementation via EM

**Script:** `P2/gmm.py`

The GMM is implemented from scratch using NumPy only (no sklearn). Key design decisions:

### Initialisation — K-means++
Centers are seeded using the K-means++ strategy: the first center is chosen uniformly at random; each subsequent center is sampled with probability proportional to the squared distance from the nearest already-chosen center. This gives a well-spread initialization that reduces the chance of poor local optima, with O(N·K) cost.

### E-step (Expectation)
For each data point **x**_n and component k, the log-responsibility is:

```
log r_{nk} = log π_k + log N(x_n | μ_k, Σ_k)
```

where log N is computed via the Mahalanobis distance and log-determinant of the covariance. The responsibilities are then normalised using the **log-sum-exp** trick to avoid numerical underflow:

```
log r_{nk} ← log r_{nk} − log Σ_j exp(log r_{nj})
```

The total log-likelihood is the sum of the log normalisation constants over all N points.

### M-step (Maximisation)
Given responsibilities r_{nk} = exp(log r_{nk}):

- **Weights:** π_k = (Σ_n r_{nk}) / N  
- **Means:** μ_k = (Σ_n r_{nk} x_n) / N_k  where N_k = Σ_n r_{nk}  
- **Covariances:** Σ_k = (Σ_n r_{nk} (x_n − μ_k)(x_n − μ_k)ᵀ) / N_k  

A small diagonal regularisation (`reg_covar = 1e-4`) is added to each Σ_k to ensure positive-definiteness.

### Convergence
EM iterates until |ΔLL| < `tol` or `max_iter` is reached.

---

## Part (b): MFCC Features + GMM Fitting

**Script:** `P2/mfcc_gmm.py`

### Feature extraction

- **MFCC:** 13 coefficients, FFT size 256 (32 ms at 8 kHz), hop 128 (50 % overlap), 26 mel filters
- All five recordings combined → **5509 frames × 13 dims**
- Features standardised (zero mean, unit variance per dimension) before fitting

### Fitting procedure

For each K ∈ {1, 2, 4, 8, 16, 32, 64}:
- 3 random restarts (K-means++ seeding) to avoid local optima
- Best run (highest log-likelihood) selected
- max iterations: 200, convergence tolerance: 1 × 10⁻⁴

### Results

| K (mixtures) | Total log-likelihood | ΔLL vs previous K |
|:---:|---:|---:|
| 1  | −97,206.41 | — |
| 2  | −67,700.65 | +29,505.76 |
| 4  | −59,786.13 | +7,914.52 |
| 8  | −51,629.06 | +8,157.07 |
| 16 | −45,213.67 | +6,415.40 |
| 32 | −41,435.21 | +3,778.46 |
| 64 | −34,404.36 | +7,030.85 |

### Plot

![GMM log-likelihood vs K](../P2/images/p2b_gmm_loglikelihood.png)

### Analysis and Comments

**Monotonic increase:** Log-likelihood increases consistently with K. This is expected — more mixture components give the model more flexibility to fit the data, so training likelihood can only improve (no regularisation penalty here).

**Largest gain at K=1→2:** The jump of +29,506 from K=1 to K=2 is by far the biggest improvement. A single Gaussian is a very poor model for speech MFCC features, which are multimodal (voiced vs. unvoiced segments, different phonemes). Adding a second Gaussian captures this coarse bimodal structure.

**Diminishing returns:** Gains decrease overall as K grows (29,506 → 7,915 → 8,157 → 6,415 → 3,778), though the K=32→64 jump (+7,031) is slightly larger than K=16→32 (+3,778), suggesting the model has not yet fully saturated at K=64 for this dataset size.

**Practical K selection:** Without a held-out validation set or a penalised criterion (BIC/AIC), training log-likelihood alone cannot identify an "optimal" K — it will always prefer larger models. For 5509 frames of 13-dim MFCCs, a K in the range **16–32** is a common practical choice for speaker/phoneme modelling (balance between expressiveness and overfitting risk).

**Convergence observation:** K=1 and K=2 converge quickly (3 and 40 iters). Larger K values hit the 200-iteration limit, indicating they have not fully converged — longer training or better initialisation (e.g. from a smaller-K solution) would yield higher likelihoods for K=16, 32, 64.

---

## Part (c): Phonetic Inspection of K=32 GMM Components

**Script:** `P2/p2c_phonetic_analysis.py`

### Methodology

The only annotated sample is `sample_2_WH720N_headphones_with_ANC.wav`, which has a Praat TextGrid with a **phoneme tier** ("phenome") containing 108 IPA-labelled intervals. The procedure:

1. Load the K=32 GMM from `gmm32_model.npz`.
2. Extract 13-dim MFCCs from `sample_2` at 8 kHz (1166 frames).
3. Assign each frame its IPA phoneme label using the frame's centre timestamp.
4. Compute GMM responsibilities and **hard-assign** each frame to its dominant component.
5. For each component, tally the distribution over 6 broad phonetic classes:
   **silence**, **vowel**, **fricative**, **stop**, **nasal**, **approximant**.

IPA symbols from the TextGrid were mapped to broad classes (e.g., ɑ/ɛ/ɔ/u/ɪ/ə → vowel; ˢ/ʃ/ʒ/ʄ/s/z → fricative; p/t/k/ʈ → stop; ɹ/l/r → approximant; m/n/ŋ → nasal).

### Results

**Class coverage across 32 components:**

| Dominant class | # components | Component IDs |
|---|:---:|---|
| Silence/pause | 10 | C4, C5, C8, C10, C13, C15, C16, C21, C26, C29 |
| Vowel | 13 | C2, C6, C9, C14, C17, C20, C23, C24, C25, C27, C28, C30, C31 |
| Fricative | 3 | C1, C3, C18 |
| Stop | 2 | C12, C22 |
| Nasal | 0 | — |
| Approximant | 0 | — |
| Other | 4 | C0, C7, C11, C19 |

**Selected component highlights:**

| Component | Dominant class | Purity | Top symbols |
|---|---|:---:|---|
| C1 | fricative | 73 % | ˢ z s ʈ |
| C5 | silence | 62 % | (pause regions) |
| C8 | silence | 54 % | (pause regions) |
| C9 | vowel | 80 % | ʊ ko o |
| C20 | vowel | 58 % | ʈu ʊ u |
| C25 | vowel | 67 % | ʈɔ ɔ o |
| C27 | vowel | 57 % | æ ɑ ðə |

### Visualisations

![Component vs phonetic class heatmap](../P2/images/p2c_component_phoneme_heatmap.png)

![Component mean MFCC vectors](../P2/images/p2c_component_means.png)

### Analysis: Do components correspond to phonetic classes?

**Partial yes — the GMM shows meaningful structure, but not clean one-class separation:**

- **Silence is well-captured:** 10 out of 32 components dominate on silence/pause frames. The GMM dedicates ~31% of its capacity to silence — reasonable given that pauses between words are spectrally distinct and consistent.

- **Vowels are distributed across 13 components:** No single "vowel component" exists; instead, different vowel qualities (back/front, open/close: ɑ, ɔ, ʊ, æ, ɛ, etc.) are captured by separate components. This is consistent with the known multimodal structure of vowel formants in MFCC space.

- **Fricatives are partially clustered:** C1 achieves 73% fricative purity (dominated by ˢ, z, s). Fricatives are spectrally distinct (broadband noise) so they naturally group together.

- **Stops are sparse:** Only 2 components show stop dominance, likely because stop bursts are short-duration and their frames are few in the training data relative to steady-state sounds.

- **Nasals and approximants have no dominant components:** These classes may be absorbed into neighbouring vowel or silence components, as their MFCC profiles overlap with both.

- **"Other" components:** Several components (C0, C7, C11, C19) are dominated by symbols that don't fit neatly into the standard IPA classification (e.g., ɭ, ᴊ, ɖ, ɓ — rare or language-specific sounds in the transcription). These likely capture unique spectral signatures of individual words.

**Conclusion:** The 32-component GMM captures coarse phonetic structure (silence vs. vowels vs. fricatives) without explicit phoneme supervision, which is a well-known property of GMMs in speech modelling. However, 32 components cannot fully resolve all phoneme classes — many components are "mixed", reflecting that MFCC space does not perfectly separate all phonetic categories.

---

## Part (d): Average Log-Likelihood — Same vs. Different Speaker

**Script:** `P2/p2d_likelihood.py`

### Setup

| | Path | Description |
|---|---|---|
| **sample-A** | `audio/R1.wav` (Assignment 2) | Same speaker as training data; new utterance (speech + background music), 44.1 kHz → resampled to 8 kHz |
| **sample-B** | `P2/audio/sample_B.wav` | TTS (speech-dispatcher system voice) — a clearly different voice/speaker not present in training |

Both samples are processed identically: downmix to mono → resample to 8 kHz → 13-dim MFCC → normalise using training statistics → score under K=32 GMM.

### Results

| Sample | Frames | Duration | Total LL | **Avg LL / frame** |
|---|:---:|:---:|---:|---:|
| sample-A (same speaker) | 3017 | 48.3 s | −48,473 | **−16.07** |
| sample-B (diff. speaker) | 1840 | 29.4 s | −40,095 | **−21.79** |

### Analysis

**sample-A has a higher (less negative) average log-likelihood than sample-B**, confirming that the GMM assigns higher probability to speech from the same speaker it was trained on.

- **Why sample-A scores higher:** The five training samples were all recorded by the same person. The GMM has learned the specific spectral envelope, speaking rate, and formant patterns characteristic of that speaker's voice. sample-A, recorded by the same person (R1.wav), falls naturally within the distribution the GMM has modelled, so each frame receives high probability under the mixture.

- **Why sample-B scores lower:** The TTS voice has different fundamental frequency, formant locations, and spectral envelope than the human training speaker. Frames from sample-B land in low-density regions of the GMM, yielding lower per-frame log-probability (−21.79 vs −16.07, a difference of ~5.7 dB in log scale).

- **Practical significance:** This ~5.7 nats/frame gap is the basis for **speaker verification / identification** systems. A GMM trained on a target speaker is used as a scoring function — samples from that speaker score much higher than imposters. This is exactly the UBM (Universal Background Model) framework used in classical speaker recognition.

- **Note on sample-B:** Here sample-B is a TTS voice (an extreme case of a "different speaker"). In practice, a human friend's recording would show a smaller but still significant gap, since both human voices share more acoustic properties than a human vs TTS.

import parselmouth
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. SETUP: Define paths and load objects
audio_path = "audio_sample/sample_2_WH720N_headphones_with_ANC.wav"
tg_path = "audio_sample/sample_2_WH720N_headphones_with_ANC.TextGrid"

snd = parselmouth.Sound(audio_path)
tg = parselmouth.read(tg_path)

# 2. DATA EXTRACTION: Extract Formants using the Burg Method
formants = snd.to_formant_burg()

# 3. SEGMENTATION: Access Tier 2 (phoneme) and identify vowels
# Use Praat-style call() to interact with the TextGrid
# Tier indices are 1-based (matching Praat), so Tier 2 = phoneme tier

# Vowels from your IPA helper chart
vowels_ipa = ['i','ɪ','e','ɛ','æ','ɑ','ɔ','ʊ','u','ʌ','ə','ɝ','aɪ','aʊ','eɪ','oʊ','ɔɪ']

vowel_data = []

# Get the number of intervals in Tier 2 using Praat's call() interface
num_intervals = parselmouth.praat.call(tg, "Get number of intervals...", 2)

for i in range(1, num_intervals + 1):
    label = parselmouth.praat.call(tg, "Get label of interval...", 2, i).strip()

    # Skip empty labels (unlabelled noise segments, silence, etc.)
    if label == "" or label not in vowels_ipa:
        continue

    start = parselmouth.praat.call(tg, "Get start time of interval...", 2, i)
    end = parselmouth.praat.call(tg, "Get end time of interval...", 2, i)

    # Calculate midpoint to capture the steady-state of the vowel
    t_mid = (start + end) / 2
    f1 = formants.get_value_at_time(1, t_mid)
    f2 = formants.get_value_at_time(2, t_mid)
    f3 = formants.get_value_at_time(3, t_mid)

    if not (np.isnan(f1) or np.isnan(f2)):
        vowel_data.append({"vowel": label, "f1": f1, "f2": f2, "f3": f3})

# 4. PLOTTING: Vowel Triangle Scatter Plot
df = pd.DataFrame(vowel_data)

plt.figure(figsize=(9, 9))
for i, row in df.iterrows():
    plt.scatter(row['f2'], row['f1'], s=200, color='royalblue', edgecolors='black', alpha=0.6)
    plt.text(row['f2']+25, row['f1'], row['vowel'], fontsize=14, fontweight='bold')

# LINGUISTIC STANDARD: Invert axes
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

plt.title("Acoustic Vowel Triangle (F1 vs F2)")
plt.xlabel("F2 Frequency (Hz) - Frontness")
plt.ylabel("F1 Frequency (Hz) - Height")
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig("images/P2.b.vowel_triangle.png", dpi=300)
plt.show()

print("\n--- Summary of Average Vowel Formants ---")
print(df.groupby('vowel').mean())
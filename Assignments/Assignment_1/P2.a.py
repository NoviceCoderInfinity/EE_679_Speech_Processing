import parselmouth
import numpy as np
import matplotlib.pyplot as plt

# Load audio
audio_path = "audio_sample/sample_2_WH720N_headphones_with_ANC.wav"
snd = parselmouth.Sound(audio_path)

# 1. DATA EXTRACTION: Extract Pitch (F0) as a function of time
# .to_pitch() uses the Cross-Correlation/Autocorrelation method found in Praat
pitch = snd.to_pitch()
pitch_values = pitch.selected_array['frequency']

# Filter out zero values (unvoiced segments like silence or consonants)
valid_f0 = pitch_values[pitch_values > 0]

# 2. CALCULATION: What is the average F0?
avg_f0 = np.mean(valid_f0)
print(f"--- Pitch Analysis Results ---")
print(f"Average F0: {avg_f0:.2f} Hz")

# 3. PLOTTING: Histogram of F0 over the whole recording
plt.figure(figsize=(10, 6))
plt.hist(valid_f0, bins=40, color='#2ab0ff', edgecolor='black', alpha=0.7)
plt.axvline(avg_f0, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {avg_f0:.2f}Hz')

plt.title("Distribution of Fundamental Frequency (F0)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Frame Count")
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("images/P2.a.f0_histogram.png", dpi=300)
plt.show()
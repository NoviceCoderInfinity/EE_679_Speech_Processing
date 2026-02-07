import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, freqz
import soundfile as sf

# =============================================================================
# CONFIGURATION — Your actual values from P2b and pitch analysis
# =============================================================================
Fs = 8000  # Sampling rate (given)
avg_F0 = 139.03  # Your average F0

# Formant frequencies from your P2b results
vowel_formants = {
    '/i/': [264.18, 2430.35, 2983.53],   # 'i' from your data
    '/u/': [291.95, 1051.69, 2875.12],   # 'ʊ' (as in push)
    '/a/': [362.97, 1188.76, 2842.51],   # 'ʌ' (as in fun)
}

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def build_vocal_tract_tf(F_list, B_list, Fs):
    """
    Build the cascade transfer function H[z] = H1[z] * H2[z] * H3[z].

    Each Hi[z] = 1 / [(1 - r_i e^{jθ_i} z^{-1})(1 - r_i e^{-jθ_i} z^{-1})]

    Expanding the denominator:
        = 1 - 2 r_i cos(θ_i) z^{-1} + r_i^2 z^{-2}
    """
    b_total = np.array([1.0])
    a_total = np.array([1.0])

    for Fi, Bi in zip(F_list, B_list):
        theta_i = 2 * np.pi * Fi / Fs
        r_i = np.exp(-2 * np.pi * Bi / Fs)

        a_i = [1, -2 * r_i * np.cos(theta_i), r_i ** 2]

        b_total = np.convolve(b_total, [1.0])
        a_total = np.convolve(a_total, a_i)

    return b_total, a_total


def generate_impulse_train(duration, F0, Fs):
    """Generate an impulse train with period matching F0."""
    n_samples = int(duration * Fs)
    period_samples = int(Fs / F0)
    signal = np.zeros(n_samples)
    signal[::period_samples] = 1.0
    return signal


def generate_half_wave_rectified_cosine(duration, F0, Fs):
    """Generate a half-wave rectified cosine signal at frequency F0."""
    n_samples = int(duration * Fs)
    t = np.arange(n_samples) / Fs
    cosine = np.cos(2 * np.pi * F0 * t)
    hw_rectified = np.maximum(cosine, 0)
    return hw_rectified


def plot_frequency_response(b, a, Fs, F_list, B_list, vowel_label, save_path=None):
    """Plot 10*log10(|H(ω)|^2) as a function of frequency."""
    w, h = freqz(b, a, worN=4096, fs=Fs)
    magnitude_db = 10 * np.log10(np.abs(h) ** 2 + 1e-12)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(w, magnitude_db, 'b-', linewidth=2)

    for i, Fi in enumerate(F_list):
        ax.axvline(x=Fi, color='red', linestyle='--', alpha=0.7,
                   label=f'F{i+1} = {Fi:.1f} Hz, B{i+1} = {B_list[i]:.1f} Hz')

    ax.set_title(f'Vocal Tract Frequency Response — Vowel {vowel_label}')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(r'$10\,\log_{10}\,|H(\omega)|^2$ (dB)')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_xlim([0, Fs / 2])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_waveforms(excitation, output, Fs, F0, vowel_label, exc_label, save_path=None):
    """Plot excitation and synthesized output waveforms."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    t = np.arange(len(excitation)) / Fs
    n_show = min(800, len(excitation))

    axes[0].plot(t[:n_show], excitation[:n_show], 'k')
    axes[0].set_title(f'{exc_label} (F0 = {F0:.1f} Hz)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, linestyle=':', alpha=0.7)

    axes[1].plot(t[:n_show], output[:n_show], 'b')
    axes[1].set_title(f'Synthesized Vowel {vowel_label}')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def make_filename(part, plot_title, vowel_tag):
    """Generate consistent filename: P3.{plot_title}.vowel_{vowel}.png"""
    # Sanitize title: lowercase, replace spaces with underscores
    title_clean = plot_title.lower().replace(' ', '_').replace('—', '').replace('/', '')
    return f"P3.{part}.{title_clean}.vowel_{vowel_tag}.png"


# =============================================================================
# PART (a): Transfer function and frequency response for /i/
# =============================================================================
print("=" * 60)
print("PART (a): Vocal Tract Transfer Function for /i/")
print("=" * 60)

F_i = vowel_formants['/i/']

# Random Q in [5, 10] for each formant
np.random.seed(42)  # Remove this line if you want different Q each run
Q_values = np.random.uniform(5, 10, size=3)
B_i = [F / Q for F, Q in zip(F_i, Q_values)]

print(f"\nChosen parameters for vowel /i/:")
for k in range(3):
    print(f"  F{k+1} = {F_i[k]:.2f} Hz,  Q{k+1} = {Q_values[k]:.2f},  B{k+1} = {B_i[k]:.2f} Hz")

b, a = build_vocal_tract_tf(F_i, B_i, Fs)

plot_frequency_response(b, a, Fs, F_i, B_i, '/i/',
                        save_path=make_filename("a", "frequency_response", "i"))


# =============================================================================
# PART (b): Vowel synthesis using impulse train excitation
# =============================================================================
print("\n" + "=" * 60)
print("PART (b): Vowel Synthesis — /i/ with Impulse Train")
print("=" * 60)

duration = 1.0
excitation = generate_impulse_train(duration, avg_F0, Fs)
output = lfilter(b, a, excitation)
output = output / np.max(np.abs(output)) * 0.9

sf.write("P3.b.vowel_i_impulse_train.wav", output, Fs)
print(f"Saved: P3.b.vowel_i_impulse_train.wav (F0 = {avg_F0} Hz)")

plot_waveforms(excitation, output, Fs, avg_F0, '/i/', 'Impulse Train Excitation',
               save_path=make_filename("b", "impulse_train_waveform", "i"))


# =============================================================================
# PART (c) BONUS: Half-wave rectified cosine excitation for /i/
# =============================================================================
print("\n" + "=" * 60)
print("PART (c) BONUS: /i/ with Half-Wave Rectified Cosine")
print("=" * 60)

hw_excitation = generate_half_wave_rectified_cosine(duration, avg_F0, Fs)
output_hw = lfilter(b, a, hw_excitation)
output_hw = output_hw / np.max(np.abs(output_hw)) * 0.9

sf.write("P3.c.vowel_i_halfwave_cosine.wav", output_hw, Fs)
print(f"Saved: P3.c.vowel_i_halfwave_cosine.wav")

plot_waveforms(hw_excitation, output_hw, Fs, avg_F0, '/i/',
               'Half-Wave Rectified Cosine',
               save_path=make_filename("c", "halfwave_cosine_waveform", "i"))


# =============================================================================
# PART (d) BONUS: Repeat for /u/ and /a/
# =============================================================================
print("\n" + "=" * 60)
print("PART (d) BONUS: Vowel Synthesis for /u/ and /a/")
print("=" * 60)

for vowel in ['/u/', '/a/']:
    F_v = vowel_formants[vowel]
    Q_v = np.random.uniform(5, 10, size=3)
    B_v = [F / Q for F, Q in zip(F_v, Q_v)]

    vowel_tag = vowel.strip('/')

    print(f"\n--- Vowel {vowel} ---")
    for k in range(3):
        print(f"  F{k+1} = {F_v[k]:.2f} Hz,  Q{k+1} = {Q_v[k]:.2f},  B{k+1} = {B_v[k]:.2f} Hz")

    b_v, a_v = build_vocal_tract_tf(F_v, B_v, Fs)

    # Frequency response
    plot_frequency_response(b_v, a_v, Fs, F_v, B_v, vowel,
                            save_path=make_filename("d", "frequency_response", vowel_tag))

    # Impulse train synthesis
    exc = generate_impulse_train(duration, avg_F0, Fs)
    out = lfilter(b_v, a_v, exc)
    out = out / np.max(np.abs(out)) * 0.9
    sf.write(f"P3.d.vowel_{vowel_tag}_impulse_train.wav", out, Fs)
    print(f"  Saved: P3.d.vowel_{vowel_tag}_impulse_train.wav")

    plot_waveforms(exc, out, Fs, avg_F0, vowel, 'Impulse Train',
                   save_path=make_filename("d", "impulse_train_waveform", vowel_tag))

    # Half-wave rectified cosine synthesis
    hw_exc = generate_half_wave_rectified_cosine(duration, avg_F0, Fs)
    out_hw = lfilter(b_v, a_v, hw_exc)
    out_hw = out_hw / np.max(np.abs(out_hw)) * 0.9
    sf.write(f"P3.d.vowel_{vowel_tag}_halfwave_cosine.wav", out_hw, Fs)
    print(f"  Saved: P3.d.vowel_{vowel_tag}_halfwave_cosine.wav")

    plot_waveforms(hw_exc, out_hw, Fs, avg_F0, vowel, 'Half-Wave Rectified Cosine',
                   save_path=make_filename("d", "halfwave_cosine_waveform", vowel_tag))

print("\n✅ All parts complete!")

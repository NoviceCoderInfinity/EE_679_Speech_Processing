# Assignment 1

# P1: Speech Recording and Transcription

## a) Recording different audio samples

Summary of the audio samples recorded:

### Algorithm 1: Suggested Algorithm

- Name of the File: SNR value
- `sample_1_laptop_microphone_with_blank_areas.wav`: 100dB
- `sample_1_laptop_microphone.wav`: 33.374 dB
- `sample_2_WH720N_headphones_with_ANC_with_blank_spaces.wav`: 100dB
- `sample_2_WH720N_headphones_with_ANC.wav`: 100dB

**Key Observation**: This algorithm was giving 100dB for all samples, no matter how clean or noisy they were with the exception of laptop microphone. Therefore, I went through the comments of the code provided and switched to a different algorithm

**Gemini Suggestion**: Gemini suggested that the code is more familiar with 16kHz or 8kHz samples, so I resampled the audio to that, but still the SNR was stuck at 100dB.

### Algorithm 2: A more neater and closer to original WADA SNR version ([GitHub Link](https://gist.github.com/peter-grajcar/4e4ebd8b700cf3e4e9e3aaff603e8426))

Below is a summary of different types of audio recordings, their SNR values, and how the samples were procured.

| Audio File Name                                               | Estimated SNR (dB) | Status / Observation                       | Audio Procurement Method                                                                                       |
| ------------------------------------------------------------- | -----------------: | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| sample_1_laptop_microphone.wav                                |              28.46 | Typical internal mic noise floor.          | Recorded using built-in laptop microphone                                                                      |
| sample_1_laptop_microphone_with_blank_areas.wav               |              33.02 | Slight variance due to zero-energy blocks. | Laptop microphone recording with silent segments inserted                                                      |
| sample_2_WH720N_headphones_with_ANC.wav                       |              74.98 | High quality, ANC effective.               | Sony WH-720N headphones with ANC enabled                                                                       |
| sample_2_WH720N_headphones_with_ANC_with_blank_spaces.wav     |              74.98 | Consistent results despite silence.        | ANC headphone recording with intentional silent gaps                                                           |
| sample_2_WH720N_headphones_without_ANC.wav                    |             100.00 | Algorithm saturation (original script).    | Headphone recording without ANC (untrimmed)                                                                    |
| sample_2_WH720N_headphones_without_ANC_trimmed.wav            |              42.41 | Realistic noisy environment estimate.      | Headphone recording without ANC, manually trimmed                                                              |
| sample_2_WH720N_headphones_without_ANC_trimmed_re-sampled.wav |             100.00 | Potential distribution mismatch.           | Trimmed headphone audio re-sampled to new rate                                                                 |
| sample_3_K8_wireless_microphone.wav                           |              46.51 | Base wireless performance.                 | Wireless K8 microphone recording with microphone held very close to audio source                               |
| sample_3_K8_wireless_microphone_take_2.wav                    |              56.11 | Good wireless capture quality.             | Wireless K8 microphone (second take), microphone at a moderate distance                                        |
| sample_3_K8_wireless_microphone_take_3.wav                    |              51.38 | Moderate wireless interference/noise.      | Wireless K8 microphone (third take), microphone at a moderate distance and words spoken very slowly but louder |

Additional information about the recording environment:

- Recording Software used: `Audacity 3.7.7`
- Microphone types used:
  - Laptop Microphone: `Microphone Array (2- Intel® Smart Sound Technology for Digital Microphones)`
  - Headphone Microphone: `Headset WH720N Sony Headphones`
  - Wireless Microphone: `K8 Wireless Microphone`
- Only one file `sample_2_WH720N_headphones_without_ANC_trimmed_re-sampled.wav` is recorded at 16kHz, rest all samples are recorded at 44.1kHz sampling rate
- The bit depth of all the fiels is set to be at `32 bit (float)`
- Sentences used: `List 6, first five sentences`

- Preferred audio sample for further processing: `sample_2_WH720N_headphones_with_ANC.wav`

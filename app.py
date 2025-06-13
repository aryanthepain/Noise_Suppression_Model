# Noise Reduction with Spectral Subtraction (Gradio demonstration)

# Author- Aryan Gupta, Tejas Deshmukh
# Date- 12/06/2025

# In this project we demonstrate **spectral subtraction**, a classical audio denoising method.
# **Idea**: Estimate the noise spectrum from a short “noisy‐only” or "almost-silent" segment, subtract it from every frame of the recording in the frequency domain, then reconstruct via inverse STFT.
# **Applications**: Noise reduction, speech enhancement, audio restoration, preprocessing for other different pipelines.

"""
-------------------------------------------------
"""
# Import required libraries
import numpy as np
np.complex = complex
# np.complex is a function, and complex is a class used by librosa to do the same thing, i.e, create complex numbers.
# It is just to keep librosa happy

import librosa        # audio I/O & processing
import librosa.display  # plotting
import soundfile as sf  # reading/writing WAV files
import matplotlib.pyplot as plt
from scipy.signal import stft, istft  # STFT operations

import gradio as gr # to create the app

"""
-------------------------------------------------
"""

# defining parameters

sr = 44100 # sampling rate (in Hz)

# STFT parameters
frame_len = 2048 # Number of samples that is analyzed as one data point. 2048 samples = 0.05 seconds at 44.1 kHz sampling rate
hop_len = 512    # how many samples you move forward after analyzing each frame. 512 samples = 0.0116 seconds at 44.1 kHz sampling rate

"""
-------------------------------------------------
"""

# Helper Functions

def extract_noise_profile(y, sr, duration=0.5):
    """
    Estimate an average magnitude spectrum from the first duration seconds of y.
    """
    n = int(sr * duration)
    noise = y[:n]
    _, _, Zxx = stft(noise, fs=sr, nperseg=frame_len,
                     noverlap=frame_len-hop_len, boundary=None)
    return np.mean(np.abs(Zxx), axis=1)


def spectral_subtract(y, sr, noise_spec):
    """
    Subtract noise_spec from the full signal in magnitude domain, then invert.
    """
    _, _, Zxx = stft(y, fs=sr, nperseg=frame_len,
                     noverlap=frame_len-hop_len, boundary=None)
    mag, phase = np.abs(Zxx), np.angle(Zxx)
    sub_mag = np.maximum(mag - noise_spec[:, None], 0.0)
    _, y_rec = istft(sub_mag * np.exp(1j * phase), fs=sr,
                     nperseg=frame_len, noverlap=frame_len-hop_len,
                     boundary=None)
    # Ensure same length
    y_rec = librosa.util.fix_length(y_rec, size=len(y))
    return y_rec

"""
-------------------------------------------------
"""

# Main function to enhance audio and return outputs

def enhance_audio(noisy_input):
    # Load audio file
    if isinstance(noisy_input, tuple):
        sr, y_noisy = noisy_input
        # If dtype is int16, convert to float in [-1,1]:
        if y_noisy.dtype.kind == 'i':
            y_noisy = y_noisy.astype(np.float32) / np.iinfo(y_noisy.dtype).max
    else:
        # assuming it's a filepath
        y_noisy, sr = librosa.load(noisy_input, sr=None)

    # Estimate noise profile
    noise_spec = extract_noise_profile(y_noisy, sr, duration=0.5)

    # Apply spectral subtraction
    y_denoised = spectral_subtract(y_noisy, sr, noise_spec)
    
    # Dummy data insertion
    margin_ms = 5  # milliseconds to replace at start
    n_dummy = int(margin_ms * 1e-3 * sr)
    # Overwrite first n_dummy samples with zeros
    y_denoised[:n_dummy] = 0
    
    # Creating comparision plot
    fig = plt.figure(figsize=(14, 8))

    # Waveforms
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y_noisy, sr=sr, alpha=0.6)
    plt.title('Original Noisy Waveform')

    plt.subplot(2, 1, 2)
    librosa.display.waveshow(y_denoised, sr=sr, color='r', alpha=0.6)
    plt.title('Denoised Waveform')

    plt.tight_layout()
    
    return (sr , y_denoised), fig

"""
-------------------------------------------------
"""

# Creating Gradio demonstration
iface1 = gr.Interface(
    fn=enhance_audio,
    inputs=gr.Audio(sources=['microphone','upload'], show_download_button=True, format='wav'),
    outputs=[gr.Audio(type="numpy"), gr.Plot()],
    theme='earneleh/paris',
    examples=['./noisy_sample_1.wav','./noisy_sample_2.wav']
)
iface1.launch()
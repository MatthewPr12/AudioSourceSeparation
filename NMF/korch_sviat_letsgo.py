import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import wave

def signal_to_wav(signal, framerate, filename):
    # save numpy array as wave file
    signal = signal.astype(np.int16)
    wave_file = wave.open(filename, 'wb')
    wave_file.setnchannels(1)
    wave_file.setsampwidth(2)
    wave_file.setframerate(framerate)
    wave_file.writeframes(signal.tobytes())
    wave_file.close()

# Load the mixed audio signal
mix_file = "../AudioData/ICA_mix3.wav"
y, sr = librosa.load(mix_file, sr=None)

# Compute the short-time Fourier transform (STFT) of the mixed signal
stft_y = librosa.stft(y)

# Calculate the magnitude spectrogram
mag_y = np.abs(stft_y)

# Apply NMF to the magnitude spectrogram
n_sources = 3
nmf = NMF(n_components=n_sources, random_state=42)
W = nmf.fit_transform(mag_y)
H = nmf.components_

# Reconstruct the separated sources
sources = np.zeros((n_sources, stft_y.shape[0], stft_y.shape[1]), dtype=np.complex64)
for i in range(n_sources):
    sources[i] = (W[:, i:i+1] @ H[i:i+1, :]) * np.exp(1j * np.angle(stft_y))

# Convert the time-frequency representation back to audio and save the separated sources
for i in range(n_sources):
    y_i = librosa.istft(sources[i])
    coeff = 10_000
    y_i *= coeff
    signal_to_wav(y_i, sr, f'separated_source_{i+1}.wav')


print("Done!")
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


# Load both audio files
file1 = "../AudioData/audio_13-14_X1.wav"
file2 = "../AudioData/audio_13-14_X2.wav"

y1, sr1 = librosa.load(file1, sr=None)
y2, sr2 = librosa.load(file2, sr=None)

# Compute the short-time Fourier transform (STFT) for both signals
stft_y1 = librosa.stft(y1)
stft_y2 = librosa.stft(y2)

# Calculate the magnitude spectrograms for both signals
mag_y1 = np.abs(stft_y1)
mag_y2 = np.abs(stft_y2)

# Combine the magnitude spectrograms into a single matrix
mag_y = np.concatenate((mag_y1, mag_y2), axis=1)

# Apply NMF to the combined magnitude spectrogram
n_sources = 2
nmf = NMF(n_components=n_sources, random_state=42)
W = nmf.fit_transform(mag_y)
H = nmf.components_

# Reconstruct the separated sources
sources = np.zeros((n_sources, stft_y1.shape[0], stft_y1.shape[1]), dtype=np.complex64)
for i in range(n_sources):
    sources[i] = (W[:, i:i + 1] @ H[i:i + 1, :stft_y1.shape[1]]) * np.exp(1j * np.angle(stft_y1))
    sources[i] *= 10_000

# Convert the time-frequency representation back to audio and save the separated sources
for i in range(n_sources):
    y_i = librosa.istft(sources[i])
    # y_i *= 10_000
    signal_to_wav(y_i, sr1, f'separated_source_{i + 1}.wav')

print("Done!")

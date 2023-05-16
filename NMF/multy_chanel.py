import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import wave


def signal_to_wav(signal, framerate, filename):
    signal = signal.astype(np.int16)
    wave_file = wave.open(filename, 'wb')
    wave_file.setnchannels(1)
    wave_file.setsampwidth(2)
    wave_file.setframerate(framerate)
    wave_file.writeframes(signal.tobytes())
    wave_file.close()


file1 = "./AudioData/sounds_mixedX.wav"
file2 = "./AudioData/sounds_mixedY.wav"

file1 = "./AudioData/ICA_mix3.wav"
file2 = "./AudioData/ICA_mix3.wav"

y1, sr1 = librosa.load(file1, sr=None)
y2, sr2 = librosa.load(file2, sr=None)

stft_y1 = librosa.stft(y1)
stft_y2 = librosa.stft(y2)

mag_y1 = np.abs(stft_y1)
mag_y2 = np.abs(stft_y2)

mag_y = np.concatenate((mag_y1, mag_y2), axis=1)

n_sources = 3

nmf = NMF(n_components=n_sources)
W = nmf.fit_transform(mag_y)
H = nmf.components_

sources = np.zeros((n_sources, stft_y1.shape[0], stft_y1.shape[1]), dtype=np.complex64)
for i in range(n_sources):
    sources[i] = (W[:, i:i + 1] @ H[i:i + 1, :stft_y1.shape[1]]) * np.exp(1j * np.angle(stft_y1))
    sources[i] *= 100_000

for i in range(n_sources):
    y_i = librosa.istft(sources[i])
    # y_i *= 10_000
    signal_to_wav(y_i, sr1, f'separated_source_{i + 1}.wav')

print("Done!")

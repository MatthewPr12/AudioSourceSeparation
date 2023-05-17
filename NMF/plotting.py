import librosa
import matplotlib.pyplot as plt
import librosa.display

def plot_sounds_spectrogram(audio_sound, sr, subtitle="", color='blue'):
    fig, ax = plt.subplots(figsize=(10, 3))
    X = librosa.stft(audio_sound)
    Xdb = librosa.amplitude_to_db(abs(X))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz', ax=ax, color=color)
    ax.set(title="The sound spectogram "+subtitle, xlabel='Time [s]', ylabel='Frequency [Hz]')


def plot_sounds_waveform(audio_sound, sr, color='blue', subtitle=""):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(audio_sound, sr=sr, ax=ax, x_axis='time', color=color)
    ax.set(title='The sound waveform' + subtitle, xlabel='Time [s]')

def plot_components(reconstructed_sounds, sr):
    n = len(reconstructed_sounds)
    colors = ['r', 'g', 'b']
    fig, ax = plt.subplots(nrows=n, ncols=1, sharex=True, figsize=(11, 3*n))
    for i in range(n):
        librosa.display.waveshow(reconstructed_sounds[i], sr=sr, color=colors[i], ax=ax[i], label=f'Source {i}',
                                 x_axis='time')
        ax[i].set(xlabel='Time [s]')
        ax[i].legend()
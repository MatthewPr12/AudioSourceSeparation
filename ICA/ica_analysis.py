from sklearn.decomposition import FastICA
from scipy import signal
from scipy.io import wavfile as wf
from matplotlib import pyplot as plt
from pydub import AudioSegment
import seaborn as sns
import numpy as np
import IPython
import wave

np.random.seed(0)

sns.set(rc={'figure.figsize': (11.7, 8.27)})
np.random.seed(0)


def ica_basics():
    n_samples = 2000
    time = np.linspace(0, 8, n_samples)
    s1 = np.sin(2 * time)
    s2 = np.sign(np.sin(3 * time))
    s3 = signal.sawtooth(2 * np.pi * time)
    S = np.c_[s1, s2, s3]
    S += 0.2 * np.random.normal(size=S.shape)
    S /= S.std(axis=0)
    A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])
    X = np.dot(S, A.T)
    ica = FastICA(n_components=3)
    S_ = ica.fit_transform(X)
    fig = plt.figure()
    models = [X, S, S_]
    names = ['mixtures', 'real sources', 'predicted sources']
    colors = ['red', 'blue', 'orange']
    for i, (name, model) in enumerate(zip(names, models)):
        plt.subplot(4, 1, i + 1)
        plt.title(name)
        for sig, color in zip(model.T, colors):
            plt.plot(sig, color=color)

    fig.tight_layout()
    plt.show()

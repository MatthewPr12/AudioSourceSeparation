from scipy.io import wavfile
from FastICA import FastICA
from FOBI import FOBI
import utilities as utl
import numpy as np


def read_data(path1, path2):
    rate1, data1 = wavfile.read(path1)
    rate2, data2 = wavfile.read(path2)
    return rate1, data1, rate2, data2


def preprocess_data(data1, data2):
    # centering + scaling + whitening
    data1 = data1 - np.mean(data1)
    data1 = data1 / 32768
    data2 = data2 - np.mean(data2)
    data2 = data2 / 32768

    matrix = np.vstack([data1, data2])
    whiteMatrix = utl.whitenMatrix(matrix)

    return whiteMatrix


def apply_FastICA(X, whiteMatrix):
    vectors = []
    for i in range(0, X.shape[0]):
        vector = FastICA(X, vectors, 0.00000001)
        vectors.append(vector)

    W = np.vstack(vectors)
    S = np.dot(W, whiteMatrix)
    return S


def apply_FOBI(X, whiteMatrix):
    fobiW = FOBI(X)
    fobiS = np.dot(fobiW.T, whiteMatrix)
    return fobiS


def plot_separated_signals(S, rate1, label, result_path1, result_path2):
    # Plot the separated sound signals
    utl.plotSounds([S[0], S[1]], ["1", "2"], rate1, label)

    # Write the separated sound signals, 5000 is multiplied so that signal is audible
    wavfile.write(result_path1, rate1, 5000 * S[0].astype(np.int16))
    wavfile.write(result_path2, rate1, 5000 * S[1].astype(np.int16))


def turn_stereo_to_mono(data, ):
    left_channel = data[:, 0]
    right_channel = data[:, 1]
    mono_data = np.mean([left_channel, right_channel], axis=0)
    return mono_data


def main():
    path1 = '../AudioData/talk_music1.wav'
    path2 = '../AudioData/talk_music2.wav'
    name1 = path1.split("/")[-1].split(".")[0]
    name2 = path2.split("/")[-1].split(".")[0]
    rate1, data1, rate2, data2 = read_data(path1, path2)

    whiteMatrix = preprocess_data(data1, data2)
    X = whiteMatrix
    S = apply_FastICA(X, whiteMatrix)
    fobiS = apply_FOBI(X, whiteMatrix)
    plot_separated_signals(S, rate1, name1 + '_' + name2 + '_' + 'separated', "./results/" + name1 + "_res.wav",
                           "./results/" + name2 + "_res.wav")
    plot_separated_signals(fobiS, rate1, name1 + '_' + name2 + '_' + 'FOBI_separated',
                           "./results/" + name1 + "FOBI_res.wav",
                           "./results/" + name2 + "FOBI_res.wav")


if __name__ == "__main__":
    main()

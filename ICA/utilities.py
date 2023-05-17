import numpy as np
import matplotlib.pyplot as plt


def whitenMatrix(matrix):
    """
    The transformation for the matrix X is given by E*D^(-1/2)*transpose(E)*X
    Where D is a diagonal matrix containing eigen values of covariance matrix of X
    E is the matrix containing eigen vectors of covariance matrix of X
    """
    # Covariance matrix is approximated by this
    covMatrix = np.dot(matrix, matrix.T) / matrix.shape[1]

    # Doing the eigen decomposition of covariance matrix of X
    eigenValue, eigenVector = np.linalg.eigh(covMatrix)
    # Making a diagonal matrix out of the array eigenValue
    diagMatrix = np.diag(eigenValue)
    # Computing D^(-1/2)
    invSqrRoot = np.sqrt(np.linalg.pinv(diagMatrix))
    # Final matrix which is used for transformation
    whitenTrans = np.dot(eigenVector, np.dot(invSqrRoot, eigenVector.T))
    # whiteMatrix is the matrix we want after all the required transformation
    # To verify, compute the covvariance matrix, it will be approximately identity
    whiteMatrix = np.dot(whitenTrans, matrix)


    return whiteMatrix


def plotSounds(sound_list, name_list, samplerate, path, toSave=False):
    """Plots the sounds as a time series data"""

    times = np.arange(len(sound_list[0])) / float(samplerate)

    fig = plt.figure(figsize=(15, 4))
    imageCoordinate = 100 + 10 * len(sound_list) + 1
    i = 0

    for sound in sound_list:
        fig.add_subplot(imageCoordinate)
        plt.fill_between(times, sound, color='k')
        plt.xlim(times[0], times[-1])
        plt.title(name_list[i])
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')
        # plt.axis("off")
        plt.plot(sound)

        imageCoordinate += 1
        i += 1

    if toSave:
        plt.savefig("./plots/sounds/" + path + ".png", bbox_inches='tight')
    plt.show()

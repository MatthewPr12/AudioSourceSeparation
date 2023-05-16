import numpy as np
import librosa
import matplotlib.pyplot as plt
import IPython.display as ipd
import sklearn

matrix_M = np.array([[1,2,3],[4,5,6],[7,8,9]])
import numpy as np
X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
from sklearn.decomposition import NMF
model = NMF(n_components=2, init='random', random_state=0)
W = model.fit_transform(X)
H = model.components_


def divergence(V,W,H):
    return 1/2*np.linalg.norm(W@H-V)


def NMF(V, S, threshold = 0.05, MAXITER = 5000): 
        
    counter = 0
    cost_function = []
    beta_divergence = 1
    
    K, N = np.shape(V)
    
    W = np.abs(np.random.normal(loc=0, scale = 2.5, size=(K,S)))    
    H = np.abs(np.random.normal(loc=0, scale = 2.5, size=(S,N)))

    while beta_divergence >= threshold and counter <= MAXITER:
        
        H *= (W.T@V)/(W.T@(W@H) + 10e-10)
        W *= (V@H.T)/((W@H)@H.T + 10e-10)
        
        beta_divergence =  divergence(V,W,H)
        cost_function.append( beta_divergence )
        counter += 1
       
    return W,H, cost_function

X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])

W, H, cost_function = NMF(X, 2, threshold = 0.05, MAXITER = 50000)

print(W@H)


from sklearn.decomposition import NMF
model = NMF(n_components=2, init='random', random_state=0)
W = model.fit_transform(X)
H = model.components_

print(W@H)
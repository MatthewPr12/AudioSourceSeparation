import numpy as np

EPS=1e-12
THRESHOLD=1e+12

class FastMNMF():
    """
    Reference: "Fast Multichannel Source Separation Based on Jointly Diagonalizable Spatial Covariance Matrices"
    """
    def __init__(self, n_basis=10, n_sources=None):
        self.n_basis = n_basis
        self.n_sources = n_sources
        self.input = None

    def __call__(self, input, iteration=100):
        self.input = input

        self.n_channels, self.n_bins, self.n_frames = self.input.shape

        if self.n_sources is None:
            self.n_sources = self.n_channels

        G = np.ones((self.n_sources, self.n_bins, self.n_channels)) * 1e-2
        for m in range(self.n_channels):
            G[m % self.n_sources, :, m] = 1

        self.basis = np.random.rand(self.n_sources, self.n_bins, self.n_basis)    
        self.activation = np.random.rand(self.n_sources, self.n_basis, self.n_frames)
        self.diagonalizer = np.tile(np.eye(self.n_channels, dtype=np.complex128), (self.n_bins, 1, 1))
        self.spatial_covariance = G

        for idx in range(iteration):
            self.update_NMF()
            self.update_SCM()
            self.update_diagonalizer()

        return self.separate(self.input)
         

    def update_NMF(self):
        X = self.input.transpose(1, 2, 0)
        g = self.spatial_covariance
        Q = self.diagonalizer
        W, H = self.basis, self.activation

        QX = np.sum(Q[:, np.newaxis, :, :] * X[:, :, np.newaxis, :], axis=3) # (self.n_bins, n_channels, n_channels) (self.n_bins, n_frames, n_channels) -> (self.n_bins, n_frames, n_channels)
        x_tilde = np.abs(QX)**2


        Lambda = W @ H
        R = np.sum(Lambda[..., np.newaxis] * g[:, :, np.newaxis], axis=0) # (self.n_bins, n_frames, n_channels)
        R[R < EPS] = EPS
        xR = x_tilde / (R ** 2)
        gxR = np.sum(g[:, :, np.newaxis] * xR[np.newaxis], axis=3)
        gR = np.sum(g[:, :, np.newaxis] / R[np.newaxis], axis=3)

        numerator = np.sum(H[:, np.newaxis, :, :] * gxR[:, :, np.newaxis], axis=3)
        denominator = np.sum(H[:, np.newaxis, :, :] * gR[:, :, np.newaxis], axis=3)
        denominator[denominator < EPS] = EPS
        W = W * np.sqrt(numerator / denominator)

        # update H
        Lambda = W @ H
        R = np.sum(Lambda[...,np.newaxis] * g[:, :, np.newaxis], axis=0) # (self.n_bins, n_frames, n_channels)
        R[R < EPS] = EPS
        xR = x_tilde / (R ** 2)
        gxR = np.sum(g[:, :, np.newaxis] * xR[np.newaxis], axis=3)
        gR = np.sum(g[:, :, np.newaxis] / R[np.newaxis], axis=3)

        numerator = np.sum(W[:, :, :, np.newaxis] * gxR[:, :, np.newaxis], axis=1)
        denominator = np.sum(W[:, :, :, np.newaxis] * gR[:, :, np.newaxis], axis=1)
        denominator[denominator < EPS] = EPS
        H = H * np.sqrt(numerator / denominator)

        self.basis, self.activation = W, H
    
    def update_SCM(self):
        g = self.spatial_covariance
        W, H = self.basis, self.activation
        X = self.input.transpose(1, 2, 0)
        Q = self.diagonalizer

        Lambda = W @ H

        R = np.sum(Lambda[..., np.newaxis] * g[:, :, np.newaxis], axis=0)
        QX = np.sum(Q[:, np.newaxis, :, :] * X[:, :, np.newaxis, :], axis=3) # (self.n_bins, n_channels, n_channels) (self.n_bins, n_frames, n_channels) -> (self.n_bins, n_frames, n_channels)
        x_tilde = np.abs(QX)**2

        R[R < EPS] = EPS
        xR = x_tilde / (R ** 2)
        
        A = np.sum(Lambda[..., np.newaxis] * xR[np.newaxis], axis=2) # (n_sources, n_bins, n_frames, n_channels)
        B = np.sum(Lambda[..., np.newaxis] / R[np.newaxis], axis=2)
        B[B < EPS] = EPS
        g = g * np.sqrt(A / B)

        self.spatial_covariance = g
    
    def update_diagonalizer(self):
        X = self.input.transpose(1, 2, 0)
        Q = self.diagonalizer
        g = self.spatial_covariance
        XX = X[:, :, :, np.newaxis] @ X[:, :, np.newaxis, :].conj()

        W, H = self.basis, self.activation
        Lambda = W @ H

        R = np.sum(Lambda[..., np.newaxis] * g[:, :, np.newaxis], axis=0)
        R[R < EPS] = EPS
        E = np.eye(self.n_channels)
        E = np.tile(E, reps=(self.n_bins, 1, 1)) # (self.n_bins, n_channels, n_channels)

        for channel_idx in range(self.n_channels):
            q_m_Hermite = Q[:, channel_idx, :]
            V = (XX / R[:, :, channel_idx, np.newaxis, np.newaxis]).mean(axis=1)
            QV = Q @ V
            condition = np.linalg.cond(QV) < THRESHOLD # (self.n_bins,)
            condition = condition[:,np.newaxis] # (self.n_bins, 1)
            e_m = E[:, channel_idx, :]
            q_m = np.linalg.solve(QV, e_m)
            qVq = q_m.conj()[:, np.newaxis, :] @ V @ q_m[:, :, np.newaxis]
            denominator = np.sqrt(qVq[...,0])
            denominator[denominator < EPS] = EPS
            # if condition number is too big, only `denominator[denominator < eps] = eps` may diverge of cost function.
            q_m_Hermite = np.where(condition, q_m.conj() / denominator, q_m_Hermite)
            Q[:, channel_idx, :] = q_m_Hermite
        self.diagonalizer = Q
    
    
    def separate(self, input):
        X = input.transpose(1, 2, 0)
        Q = self.diagonalizer # (self.n_bins, n_channels, n_channels)
        g = self.spatial_covariance

        W, H = self.basis, self.activation
        Lambda = W @ H
        
        LambdaG = Lambda[..., np.newaxis] * g[:, :, np.newaxis, :] # (n_sources, n_bins, n_frames, n_channels)
        y_tilde = np.sum(LambdaG, axis=0) # (self.n_bins, n_frames, n_channels)
        Q_inverse = np.linalg.inv(Q) # (self.n_bins, n_channels, n_channels)
        QX = np.sum(Q[:, np.newaxis, :] * X[:, :, np.newaxis], axis=3) # (self.n_bins, n_frames, n_channels)
        y_tilde[y_tilde < EPS] = EPS
        QXLambdaGy = QX * (LambdaG / y_tilde) # (n_sources, n_bins, n_frames, n_channels)
        
        x_hat = np.sum(Q_inverse[:, np.newaxis, :, :] * QXLambdaGy[:, :, :, np.newaxis, :], axis=4) # (n_sources, n_bins, n_frames, n_channels)
        x_hat = x_hat.transpose(0, 3, 1, 2) # (n_sources, n_channels, n_bins, n_frames)
        
        return x_hat[:, 0, :, :]
    
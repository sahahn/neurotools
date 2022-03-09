from sklearn.base import BaseEstimator
import numpy as np


class OrthRegression(BaseEstimator):
    
    def __init__(self):
        pass
    
    def fit(self, X, y):
        
        # Work with 1D
        x = np.squeeze(X)
        
        # Concat
        Z  = np.c_[x, y]
        
        # Compute SVD
        meanZ = np.tile(Z.mean(axis=0), (Z.shape[0], 1))
        V = np.linalg.svd(Z - meanZ)[2]

        # Slope
        self.a = -V[1][0] / V[1][1]

        # Coef
        self.b = np.mean(np.matmul(Z, V[1]) / V[1, 1])

        # Save normal for use w/ get orth dist
        self.normal = V[1]

        return self
        
    def predict(self, X):
        
        # Work with 1D x
        x = np.squeeze(X)
        
        # Get resid
        resid = (x * self.a) + self.b
        
        # Return as len(x) x 1
        return np.expand_dims(resid, -1)
        
    def get_orth_dist(self, X, y):

        # Get z
        x = np.squeeze(X)
        Z  = np.c_[x, y]
        meanZ = np.tile(Z.mean(axis=0), (Z.shape[0], 1))

        # Compute orth distance
        return np.abs(np.matmul((Z - meanZ), self.normal))
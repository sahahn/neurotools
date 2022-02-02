"""
PCA dimension reduction on multiple subjects.
This is a good initialization method for ICA.
"""
import numpy as np
from joblib import Memory
from sklearn.utils.extmath import randomized_svd

from .base import BaseDecomposition

class MultiPCA(BaseDecomposition):

    def __init__(self, n_components=20,
                 do_cca=True,
                 random_state=None,
                 n_jobs=1,
                 verbose=0):
    
        self.n_components = n_components
        self.do_cca = do_cca

        BaseDecomposition.__init__(self, n_components=n_components,
                                   random_state=random_state,
                                   n_jobs=n_jobs,
                                   verbose=verbose)

    def _raw_fit(self, data):
        """Helper function that directly process unmasked data"""

        if self.do_cca:
            S = np.sqrt(np.sum(data ** 2, axis=1))
            S[S == 0] = 1
            data /= S[:, np.newaxis]

        components_, self.variance_, _ = randomized_svd(
            data.T, n_components=self.n_components,
            transpose=True,
            random_state=self.random_state, n_iter=3)
        
        if self.do_cca:
            data *= S[:, np.newaxis]
        self.components_ = components_.T
    
        return components_
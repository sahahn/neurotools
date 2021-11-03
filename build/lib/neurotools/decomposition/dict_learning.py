"""
Dictionary learning estimator: Perform a map learning algorithm by learning
a temporal dense dictionary along with sparse spatial loadings, that
constitutes output maps
"""

# Author: Arthur Mensch
# License: BSD 3 clause

import warnings

import numpy as np
from sklearn.decomposition import dict_learning_online
from sklearn.linear_model import Ridge

from .base import BaseDecomposition
from .canica import CanICA

# check_input=False is an optimization available in sklearn.
sparse_encode_args = {'check_input': False}

def _compute_loadings(components, data):
    ridge = Ridge(fit_intercept=None, alpha=1e-8)
    ridge.fit(components.T, np.asarray(data.T))
    loadings = ridge.coef_.T

    S = np.sqrt(np.sum(loadings ** 2, axis=0))
    S[S == 0] = 1
    loadings /= S[np.newaxis, :]
    return loadings


class DictLearning(BaseDecomposition):

    def __init__(self,
                 n_components=20,
                 n_epochs=1,
                 alpha=10,
                 reduction_ratio='auto',
                 dict_init=None,
                 random_state=None,
                 batch_size=20,
                 method="cd",
                 n_jobs=1,
                 verbose=0):

        BaseDecomposition.__init__(self, n_components=n_components,
                                   random_state=random_state,
                                   n_jobs=n_jobs,
                                   verbose=verbose)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.method = method
        self.alpha = alpha
        self.reduction_ratio = reduction_ratio
        self.dict_init = dict_init

    def _init_dict(self, data):

        if self.dict_init is not None:
            components = self.dict_init

        else:
            canica = CanICA(n_components=self.n_components,
                            do_cca=True,
                            threshold=float(self.n_components),
                            n_init=1,
                            random_state=self.random_state,
                            n_jobs=self.n_jobs, verbose=self.verbose)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                canica._raw_fit(data)

            components = canica.components_

        S = (components ** 2).sum(axis=1)
        S[S == 0] = 1
        components /= S[:, np.newaxis]
        self.components_init_ = components

    def _init_loadings(self, data):
        self.loadings_init_ = _compute_loadings(self.components_init_, data)

    def _raw_fit(self, data):
        """Helper function that directly process unmasked data
        Parameters
        ----------
        data : ndarray,
            Shape (n_samples, n_features)
        """
        if self.verbose:
            print('[DictLearning] Learning initial components')
        self._init_dict(data)

        _, n_features = data.shape

        if self.verbose:
            print('[DictLearning] Computing initial loadings')
        self._init_loadings(data)

        dict_init = self.loadings_init_

        n_iter = ((n_features - 1) // self.batch_size + 1) * self.n_epochs

        if self.verbose:
            print('[DictLearning] Learning dictionary')

        self.components_, _ = dict_learning_online(
            data.T, self.n_components, alpha=self.alpha, n_iter=n_iter,
            batch_size=self.batch_size, method=self.method,
            dict_init=dict_init, verbose=max(0, self.verbose - 1),
            random_state=self.random_state, return_code=True, shuffle=True,
            n_jobs=1)
        self.components_ = self.components_.T

        # Unit-variance scaling
        S = np.sqrt(np.sum(self.components_ ** 2, axis=1))
        S[S == 0] = 1
        self.components_ /= S[:, np.newaxis]

        # Flip signs in each composant so that positive part is l1 larger
        # than negative part. Empirically this yield more positive looking maps
        # than with setting the max to be positive.
        for component in self.components_:
            if np.sum(component > 0) < np.sum(component < 0):
                component *= -1

        return self
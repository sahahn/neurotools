"""
Base class for decomposition estimators, utilities for masking and dimension
reduction of group data
"""
import numpy as np
from scipy import linalg
from math import ceil
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Memory, Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd, svd_flip

from ..transform.rois import SurfMaps

import warnings
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    from nilearn.signal import _row_sum_of_squares


def fast_svd(X, n_components, random_state=None):
    """ Automatically switch between randomized and lapack SVD (heuristic
        of scikit-learn).
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data to decompose
    n_components : integer
        The order of the dimensionality of the truncated SVD
    random_state : int or RandomState, optional
        Pseudo number generator state used for random sampling.
    Returns
    -------
    U : array, shape (n_samples, n_components)
        The first matrix of the truncated svd
    S : array, shape (n_components)
        The second matrix of the truncated svd
    V : array, shape (n_components, n_features)
        The last matrix of the truncated svd
    """
    random_state = check_random_state(random_state)

    # Small problem, just call full PCA
    if max(X.shape) <= 500:
        svd_solver = 'full'
    elif n_components >= 1 and n_components < .8 * min(X.shape):
        svd_solver = 'randomized'
    # This is also the case of n_components in (0,1)
    else:
        svd_solver = 'full'

    # Call different fits for either full or truncated SVD
    if svd_solver == 'full':
        U, S, V = linalg.svd(X, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        U, V = svd_flip(U, V)
        # The "copy" are there to free the reference on the non reduced
        # data, and hence clear memory early
        U = U[:, :n_components].copy()
        S = S[:n_components]
        V = V[:n_components].copy()
    else:
        n_iter = 'auto'

        U, S, V = randomized_svd(X, n_components=n_components,
                                 n_iter=n_iter,
                                 flip_sign=True,
                                 random_state=random_state)
    return U, S, V


def mask_and_reduce(imgs,
                    reduction_ratio='auto',
                    n_components=None,
                    random_state=None,
                    n_jobs=1):

    if reduction_ratio == 'auto':
        if n_components is None:
            # Reduction ratio is 1 if
            # neither n_components nor ratio is provided
            reduction_ratio = 1
    else:
        if reduction_ratio is None:
            reduction_ratio = 1
        else:
            reduction_ratio = float(reduction_ratio)
        if not 0 <= reduction_ratio <= 1:
            raise ValueError('Reduction ratio should be between 0. and 1.,'
                             'got %.2f' % reduction_ratio)

    if reduction_ratio == 'auto':
        n_samples = n_components
        reduction_ratio = None
    else:
        # We'll let _mask_and_reduce_single decide on the number of
        # samples based on the reduction_ratio
        n_samples = None

    data_list = Parallel(n_jobs=n_jobs)(
        delayed(_mask_and_reduce_single)(
            img.T,
            reduction_ratio=reduction_ratio,
            n_samples=n_samples,
            random_state=random_state
        ) for img in imgs)
    
    
    return np.vstack(data_list)

def _mask_and_reduce_single(img,
                            reduction_ratio=None,
                            n_samples=None,
                            random_state=None):
    """Utility function for multiprocessing from MaskReducer"""
    random_state = check_random_state(random_state)

    data_n_samples = img.shape[0]
    if reduction_ratio is None:
        assert n_samples is not None
        n_samples = min(n_samples, data_n_samples)
    else:
        n_samples = int(ceil(data_n_samples * reduction_ratio))

    U, S, V = fast_svd(img, n_samples, random_state=random_state)
    U = U.T.copy()
    U = U * S[:, np.newaxis]
    return U


class BaseDecomposition(BaseEstimator, TransformerMixin):
    
    def __init__(self, n_components=20,
                 random_state=None,
                 n_jobs=1,
                 verbose=0):

        self.n_components = n_components
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, imgs, y=None):
        
        data = mask_and_reduce(
                    imgs,
                    reduction_ratio='auto',
                    n_components=self.n_components,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs)

        self._raw_fit(data)

        # Create and fit NiftiMapsMasker for transform
        # and inverse_transform
        self.surf_maps_ = SurfMaps(maps=self.components_.T,
                                   strategy='ls', vectorize=False)

        return self

    def _check_components_(self):
        if not hasattr(self, 'components_'):
            raise ValueError("Object has no components_ attribute. "
                             "This is probably because fit has not "
                             "been called.")

    def transform(self, imgs):
        self._check_components_()
        return np.array([self.surf_maps_.transform(img) for img in imgs])


def explained_variance(X, components, per_component=True):
    """Score function based on explained variance
        Parameters
        ----------
        X : ndarray
            Holds single subject data to be tested against components.
        components : array-like
            Represents the components estimated by the decomposition algorithm.
        per_component : boolean, optional
            Specify whether the explained variance ratio is desired for each
            map or for the global set of components_.
            Default=True.
        Returns
        -------
        score : ndarray
            Holds the score for each subjects. score is two dimensional if
            per_component = True.
        """
    full_var = np.var(X)
    n_components = components.shape[0]
    S = np.sqrt(np.sum(components ** 2, axis=1))
    S[S == 0] = 1
    components = components / S[:, np.newaxis]
    projected_data = components.dot(X.T)
    if per_component:
        res_var = np.zeros(n_components)
        for i in range(n_components):
            res = X - np.outer(projected_data[i],
                               components[i])
            res_var[i] = np.var(res)
            # Free some memory
            del res
        return np.maximum(0., 1. - res_var / full_var)
    else:
        lr = LinearRegression(fit_intercept=True)
        lr.fit(components.T, X.T)
        res = X - lr.coef_.dot(components)
        res_var = _row_sum_of_squares(res).sum()
        return np.maximum(0., 1. - res_var / _row_sum_of_squares(X).sum())
import os
import warnings
import numpy as np
from scipy.stats import ks_2samp, gaussian_kde
from scipy.spatial.distance import jensenshannon
from .parc import merge_parc_hemis

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import itertools
import pandas as pd

from ..loading import load
from ..stats.orth import OrthRegression
from scipy.stats import pearsonr

def _get_kde_p(data, n=256):
    kde = gaussian_kde(data)
    return kde(np.linspace(np.min(data), np.max(data), n))

def _jsd_metric(x, y):

    # Get kde est
    x_p = _get_kde_p(x)
    y_p = _get_kde_p(y)

    # Then return 1 - dist
    return 1 - jensenshannon(x_p, y_p)

def _ks_metric(x,  y):
    '''1 - kd_2samp metric'''

    return 1 - ks_2samp(x, y).statistic

def _ks_demean_metric(x, y):
    
    return _ks_metric(x - np.mean(x),
                      y - np.mean(y))

def _ks_normalize_metric(x, y):
    
    return _ks_metric((x - np.mean(x)) / np.std(x),
                      (y - np.mean(y)) / np.std(y))

def gen_indv_roi_network(data, labels, metric='jsd', vectorize=False, discard_diagonal=False):
    '''This function is designed to generate a network
    of 1 - distance function between groups of different ROIs.
    

    Specifically, this method calls the function :func:`scipy.stats.ks_2samp`  or
    :func:`scipy.spatial.distance.jensenshannon` to calculate the distances
    between each collection of data points in each pair of ROIs, calculating the
    distance between the two distributions. Values are then subtracted from 1,
    such that exact distribution matches have value 1, i.e., these are the
    strongest connections. 

    Parameters
    ----------
    data : 1D numpy array
        Data at this stage must be a single dimensional
        numpy array representing the underlying data in which
        the labelled regions correspond to.

        For example, this is typically a subject's neuroimaging
        data along with corresponding label file in some ROI space.

    labels : 1D numpy array
        Labels in the form of a 1D numpy array with the same shape / len
        as data. This file should contain integer labels
        corresponding to which vertex or data elements belong
        to different ROIs.

    metric : {'jsd', 'ks', 'ks_demean', 'ks_normalize'}, optional
        The type of distance to compute between points.

        - 'jsd' : Use the 1 - Jensen Shannon distance between each rois kde estimated distributions.
        - 'ks' : Use the 1 - Kolmogorov–Smirnov distance between distributions.
        - 'ks_demean' : Same as 'ks', but with each ROI's points de-meaned.
        - 'ks_normalize' : Same as 'ks', but with each ROI's points normalized.

        ::

            default = 'jsd'

    vectorize : bool, optional
        If True, matrices are reshaped into 1D arrays and only their
        flattened lower triangular parts are returned.

        ::

            default = False

    discard_diagonal : bool, optional
        If True, the diagonal elements of the calculated matrix are
        discarded.

        ::

            default = False

    Returns
    --------
    matrix : 1/2D numpy array
        Base behavior is a two dimensional array containing the
        distances / statistics between
        every combination of ROI. If vectorize is True, then
        a 1D array is returned instead, and if discard_diagonal
        is set, then the diag will be removed as well.
    '''

    # Get correct metric function
    if metric == 'jsd':
        dist_metric = _jsd_metric
    elif metric == 'ks':
        dist_metric = _ks_metric
    elif metric == 'ks_demean':
        dist_metric = _ks_demean_metric
    elif metric == 'ks_normalize':
        dist_metric = _ks_normalize_metric
    else:
        raise RuntimeError(f'Passed metric {metric} invalid!')

    # Warn if improper input
    if discard_diagonal is True and vectorize is False:
        warnings.warn('Note: not discarding diagonal since vectorize is False.')

    # Generate the corr matrix
    n_unique = len(np.unique(labels))
    matrix = np.zeros((n_unique, n_unique))

    for i, u1 in enumerate(np.unique(labels)):
        for j, u2 in enumerate(np.unique(labels)):

            # Skip self
            if i == j:
                continue

            # If already calculated reverse, skip
            if matrix[j][i] == 0:
                matrix[i][j] = dist_metric(data[labels==u1], data[labels==u2])
                matrix[j][i] = matrix[i][j]

    # If vectorize, return just lower triangle
    if vectorize:

        # Keep diagonal or not
        k = 0
        if discard_diagonal:
            k = -1

        return matrix[np.tril_indices(matrix.shape[0], k=k)]

    # Otherwise, return full matrix
    return matrix

def _load_fs_subj_data(subj_dr, modality, parc):
    
    # Path to their thickness file
    data_lh = load(os.path.join(subj_dr, 'surf', f'lh.{modality}'))
    data_rh = load(os.path.join(subj_dr, 'surf', f'rh.{modality}'))
    data = np.concatenate([data_lh, data_rh])

    # Get merged label file
    lh_labels = os.path.join(subj_dr, 'label', f'lh.{parc}')
    rh_labels = os.path.join(subj_dr, 'label', f'rh.{parc}')
    labels = merge_parc_hemis(lh_labels, rh_labels)
    
    return data, labels

def gen_fs_subj_vertex_network(subj_dr, modality='thickness',
                               parc='aparc.a2009s.annot', metric='jsd',
                               vectorize=False, discard_diagonal=False):
    '''This function is helper function for calling
    :func:`gen_indv_roi_network`, for data organized in
    freesurfer individual subject directory style.

    Parameters
    ----------
    subj_dr : str
        The str location of the subject's freesurfer
        directory in which to generate the ks roi network.

    modality : str, optional
        The name of the modality (e.g., thickness, sulc, area, ...)
        in which to generate the network. Should
        be saved in subdirectory surf with names:
        `lh.{modality}` and `rh.{modality}`

        ::

            default = 'thickness'

    parc : str, optional
        The name of the label / parcellation file in which
        to use, as found in subdirectory label with names:
        `lh.{parc}` and `rh.{parc}`.

        These are concatenated internally with function
        :func:`merge_parc_hemis`.

        The default value is to use the destr. parcellation.

        ::

            default = 'aparc.a2009s.annot'

     metric : {'jsd', 'ks', 'ks_demean', 'ks_normalize'}, optional
        The type of distance to compute between points.

        - 'jsd' : Use the 1 - Jensen Shannon distance between each rois kde estimated distributions.
        - 'ks' : Use the 1 - Kolmogorov–Smirnov distance between distributions.
        - 'ks_demean' : Same as 'ks', but with each ROI's points de-meaned.
        - 'ks_normalize' : Same as 'ks', but with each ROI's points normalized.

        ::

            default = 'jsd'

    vectorize : bool, optional
        If True, matrices are reshaped into 1D arrays and only their
        flattened lower triangular parts are returned.

        ::

            default = False

    discard_diagonal : bool, optional
        If True, the diagonal elements of the calculated matrix are
        discarded.

        ::

            default = False

    Returns
    --------
    matrix : 1/2D numpy array
        Two dimensional array containing the
        1 - Kolmogorov–Smirnov statistic / distances between
        every combination of ROI.

        If vectorize is True, then a 1D array is returned.

    '''
    
    # Load in data
    data, labels = _load_fs_subj_data(subj_dr, modality, parc)

    # Call main function
    return gen_indv_roi_network(data, labels, metric=metric,
                                vectorize=vectorize,
                                discard_diagonal=discard_diagonal)


class GroupDifNetwork(BaseEstimator, TransformerMixin):

    _needs_fit_index = True

    def __init__(self, vectorize=True, scale_weights=False,
                 fit_only_group_index=None, pval_thresh=None,
                 passthrough=False, verbose=0):
        
        self.vectorize = vectorize
        self.scale_weights = scale_weights
        self.fit_only_group_index = fit_only_group_index
        self.pval_thresh = pval_thresh
        self.passthrough = passthrough
        self.verbose = verbose

    def proc_X(self, X, fit_index=None):
        
        # Use method from sklearn base estimator
        self._check_feature_names(X, reset=True)

        # If df, get fit_index, but convert to numpy array
        # internally, so easier to work with, i.e., consistent indexing.
        if isinstance(X, pd.DataFrame):
            fit_index = X.index
            X = np.array(X)

        return X, fit_index

    def fit(self, X, y=None, fit_index=None):

        X, fit_index  = self.proc_X(X, fit_index=fit_index)

        # More efficient to call fit transform
        # directly, but should still support calling
        # separate

        # Need to call fit transform
        # for fit, if scale weights is True
        if self.scale_weights:
            self.fit_transform(X=X, y=y, fit_index=fit_index)
        
        # Otherwise can just call base fit
        else:
            self._fit(X=X, y=y, fit_index=fit_index)
        
        return self
        
    def _fit(self, X, y=None, fit_index=None):

        if self.verbose > 1:
            print(f'Passed X shape={X.shape} to fit')

        if self.fit_only_group_index is not None:

            if fit_index is None:
                raise RuntimeError('Since using fit_only_group_index, need fit_index here!')

            # Set inds to only group
            inds = fit_index.isin(self.fit_only_group_index)
            X_s = X[inds]
        
        # Otherwise use all subjects to fit reference group
        else:
            X_s = X

        if self.verbose > 1:
            print('Fitting with shape:', X_s.shape, flush=True)
        
        # Gen each possible pair        
        self.pairs_ = list(itertools.combinations(range(X.shape[1]), 2))

        # Fit orth regression for each combination of regions
        self.estimators_ = []
        for i, j in self.pairs_:

            # If non-null pval thresh, add also if
            # base pearson pval is over threshold, add to skip mask
            if self.pval_thresh is not None:
                corr_sig = pearsonr(X_s[:, i], X_s[:, j])[1]

                # If doesn't pass, skip
                if corr_sig > self.pval_thresh:
                    self.estimators_.append(None)
                    continue
            
            # Otherwise, fit OrthRegression
            model = OrthRegression().fit(X_s[:, i], X_s[:, j])
            self.estimators_.append(model)

        
        return self

    def _proc_scale(self, X_trans, fit):

        # If scale weights is True
        if self.scale_weights:
            
            # Fit case, init scaler and fit
            if fit:

                # Fit to X_trans between 0 and 1
                self.scaler_ = MinMaxScaler(feature_range=(0, 1), clip=True)
                self.scaler_.fit(X_trans)
            
            # Always transform
            X_trans = self.scaler_.transform(X_trans)

        return X_trans
    
    def _transform(self, X, fit=False):
        
        # Go through each fitted, and get the residuals
        X_trans = []
        for estimator, ij in zip(self.estimators_, self.pairs_):

            # Just add as NaNs for all if skip
            if estimator is None:
                nans = np.empty(len(X))
                nans[:] = np.nan
                X_trans.append(nans)
                continue
            
            # Unpack
            i, j = ij

            # Get the orth distance between estimate
            # and current point - this estimate is symmetric
            X_trans.append(estimator.get_orth_dist(X[:, i], X[:, j]))
        
        # Set as # of subjects x number of edge combos
        X_trans = np.stack(X_trans, axis=-1)

        # Process if scale weights
        X_trans = self._proc_scale(X_trans, fit)

        if self.verbose > 1:
            print(f'Base transformed shape: {X_trans.shape}', flush=True)
                
        # Return as is, if default
        if self.vectorize:

            # If passthrough and vectorize, then combine
            if self.passthrough:
                return np.hstack([X_trans, X])

            return X_trans
        
        # Otherwise, cast X_trans to full network matrix
        return self._to_matrix(X, X_trans)
    
    def transform(self, X):
 
        # Base transform w/o fit
        return self._transform(X, fit=False)
    
    def _to_matrix(self, X, X_trans):
        '''Transforms to full matrix, w/ redundancy'''
        
        # Init empty network to fill, where 0's represent
        # same pair connections
        as_matrix = np.zeros((X.shape[0], X.shape[1], X.shape[1]))
        for feat, ij in enumerate(self.pairs_):
            i, j = ij
            as_matrix[:, i, j] = X_trans[:, feat]
            as_matrix[:, j, i] = X_trans[:, feat]
            
        return as_matrix
    
    def fit_transform(self, X, y=None, fit_index=None):

        X, fit_index  = self.proc_X(X, fit_index=fit_index)
        
        # Fit transform, calling base fit and base transform _transform
        return self._fit(X=X, y=y, fit_index=fit_index)._transform(X=X, fit=True)



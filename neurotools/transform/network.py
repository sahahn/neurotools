import os
import warnings
import numpy as np
from scipy.stats import ks_2samp, gaussian_kde
from scipy.spatial.distance import jensenshannon
from .parc import merge_parc_hemis
from ..loading import load

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

    metric : {'jsd', 'ks', 'ks demean', 'ks normalize'}, optional
        The type of distance to compute between points.

        - 'jsd' : Use the 1 - Jensen Shannon distance between each rois kde estimated distributions.
        - 'ks' : Use the 1 - Kolmogorov–Smirnov distance between distributions.
        - 'ks demean' : Same as 'ks', but with each ROI's points de-meaned.
        - 'ks normalize' : Same as 'ks', but with each ROI's points normalized.

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
    elif metric == 'ks demean':
        dist_metric = _ks_demean_metric
    elif metric == 'ks normalize':
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

     metric : {'jsd', 'ks', 'ks demean', 'ks normalize'}, optional
        The type of distance to compute between points.

        - 'jsd' : Use the 1 - Jensen Shannon distance between each rois kde estimated distributions.
        - 'ks' : Use the 1 - Kolmogorov–Smirnov distance between distributions.
        - 'ks demean' : Same as 'ks', but with each ROI's points de-meaned.
        - 'ks normalize' : Same as 'ks', but with each ROI's points normalized.

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
  





import os
import warnings
import numpy as np
from scipy.stats import ks_2samp
from .parc import merge_parc_hemis
from ..loading import load

def gen_ks_roi_network(data, labels, vectorize=False, discard_diagonal=False):
    '''This function is designed to generate a network
    of 1 - Kolmogorov–Smirnov distances between groups of different ROIs.

    Specifically, this method calls the function :func:`scipy.stats.ks_2samp` between
    each collection of data points in each pair of ROIs, calculating the
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
                matrix[i][j] = 1 - ks_2samp(data[labels==u1], data[labels==u2]).statistic
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
                               parc='aparc.a2009s.annot',
                               vectorize=False, discard_diagonal=False):
    '''This function is helper function for calling
    :func:`gen_ks_roi_network`, for data organized in
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
    return gen_ks_roi_network(data, labels, vectorize=vectorize,
                              discard_diagonal=discard_diagonal)
  





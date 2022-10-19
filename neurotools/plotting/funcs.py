from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.colors as colors
import matplotlib.pyplot as plt


def _collapse_data(data):
    '''Assumes data is in standard {} form.'''

    # If directly nifti image
    if isinstance(data, nib.Nifti1Image):
        return np.array(data.get_fdata()).flatten()

    # Directly ndarray case
    elif isinstance(data, np.ndarray):
        return np.array(data).flatten()

    # Single value case
    elif isinstance(data, (float, int)):
        return data

    # Init empty list
    collapsed = []

    # Handle passed as dict or list
    # of arrays
    for key in data:

        # Get as item / array
        if isinstance(data, dict):
            item = data[key]
        else:
            item = key

        # Recursive case add
        collapsed.append(_collapse_data(item))

    # Return as concat version
    return np.concatenate(collapsed)


def _trunc_cmap(cmap, minval=0.0, maxval=1.0, num=1000):

    new_cmap = colors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, num)))

    return new_cmap


def _proc_cmap(cmap, rois, symmetric_cbar, flat_data):
    
    # Keep user passed if not user passed
    if not (cmap == 'default' or cmap is None or cmap == 'auto'):
        return cmap
    
    # If plotting rois
    if rois:
        return 'prism'
    
    # If not symmetric, then just do Reds or Blues
    if symmetric_cbar is False:

        # If max value is 0 or less, use Blues
        if np.nanmax(flat_data)  <= 0:
            return 'Blues_r'

        # Otherwise, reds
        return 'Reds'

    # Last case is symmetric cbar
    return _trunc_cmap(plt.get_cmap('cold_white_hot'),
                       minval=0.05, maxval=0.95)


def _get_if_sym_cbar(data, symmetric_cbar, rois=False):
    '''Assumes data is in standard {} form.'''

    # If user passed, keep that value
    if symmetric_cbar not in ('auto', 'default'):
        return symmetric_cbar

    # If rois, default = False, so return
    if rois:
        return False

    # Get data as 1D array
    flat_data = _collapse_data(data)

    # If all positive or negative, assume false
    if np.all(flat_data >= 0) or np.all(flat_data <= 0):
        return False

    # Otherwise, assume is symmetric
    return True


def _proc_vs(data, vmin, vmax, symmetric_cbar):

    # If already set, skip
    if vmin is not None and vmax is not None:
        return vmin, vmax

    # Get data as flat array
    flat_data = _collapse_data(data)

    # Small percent of min / max to add to either end
    s = np.nanmax(np.abs(flat_data)) / 25

    # Both not set case
    if vmin is None and vmax is None:

        # If not symmetric_cbar, then these value
        # stay fixed as this
        vmin, vmax = np.nanmin(flat_data) - s, np.nanmax(flat_data) + s

        # If symmetric need to override either
        # vmin or vmax to be the opposite of the larger
        if symmetric_cbar:

            # If vmin is larger, override vmax
            if np.abs(vmin) > vmax:
                vmax = -vmin

            # If vmax is larger, override vmin
            else:
                vmin = -vmax

    # If just vmin not set
    if vmin is None:

        # If vmax set, vmin not, and symmetric cbar
        # then vmin is -vmax
        if symmetric_cbar:
            vmin = -vmax

        # Otherwise, vmin is just the min value in the data
        else:
            vmin = np.nanmin(flat_data) - s

    # If vmax not set, same cases as above but flipped
    if vmax is None:

        if symmetric_cbar:
            vmax = -vmin
        else:
            vmax = np.nanmax(flat_data) + s

    return vmin, vmax


def _get_col_names(df):

    all_cols = list(df)
    name_col, value_col = None, None

    for col in all_cols:
        try:
            df[col].astype('float')
            value_col = col
        except ValueError:
            name_col = col

    return name_col, value_col


def _data_to_dict(data):

    # If series convert to dict directly
    if isinstance(data, pd.Series):
        return data.to_dict()

    # If df
    elif isinstance(data, pd.DataFrame):

        # If only 1 col, use index as name col
        if len(list(data)) == 1:
            return data[list(data)[0]].to_dict()

        # If 2 col DF proc
        elif len(list(data)) == 2:

            as_dict = {}
            name_col, value_col = _get_col_names(data)

            for i in data.index:
                name = data[name_col].loc[i]
                as_dict[name] = data[value_col].loc[i]

            return as_dict

        # Otherwise, assume is in format of
        # feats by repeated measurements and use mean
        return data.mean().to_dict()

    elif isinstance(data, dict):
        return data

    raise RuntimeError('Invalid format passed for data, '
                       'must be dict, pd.Series or pd.DataFrame')


def _get_colors(weights, ref_weights, symmetric_cbar='auto',
                cmap='default', vmin=None, vmax=None):
    '''Idea is use ref weights to determine sym cbar, cmap, vmin, vmax
    then normalize just the base weights - since get colors is
    used on a subset of the thing to plot.'''

    # Process automatic symmetric_cbar
    symmetric_cbar = _get_if_sym_cbar(ref_weights,
                                      symmetric_cbar=symmetric_cbar,
                                      rois=False) 

    # Get cmap based on passed + if rois or not + if sym
    cmap = _proc_cmap(cmap, rois=False, symmetric_cbar=symmetric_cbar,
                      flat_data=ref_weights)
    our_cmap = get_cmap(cmap)

    # Get normalize object - should end up 0 to 1
    vmin, vmax = _proc_vs(ref_weights, vmin=vmin, vmax=vmax,
                          symmetric_cbar=symmetric_cbar)
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    return [our_cmap(norm(w)) for w in weights]


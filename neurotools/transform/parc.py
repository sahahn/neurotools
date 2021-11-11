import numpy as np
from ..loading import load

def _proc_background_label(data, background_label):
    '''Sets passed background label to 0'''

    # If None or 0, do nothing
    if background_label is None or background_label == 0:
        return data

    # If list / array-like
    if hasattr(background_label, '__iter__'):
        data[np.isin(data, background_label)] = 0

    # If non-zero single label
    else:
        data[data == background_label] = 0

    return data

def merge_parc_hemis(lh, rh, background_label=0, merge_thresh=.75):
    '''This function is designed to help merge two surface parcellations when
    saved in separate files or data arrays.

    Parameters
    -----------
    lh : str, :func:`arrays<numpy.array>` or other
        The str location or loaded numpy array-like representing
        the left hemisphere values in which to merge.

    rh : str, :func:`arrays<numpy.array>` or other
        The str location or loaded numpy array-like representing
        the right hemisphere values in which to merge.

    background_label : int, list-like of int or None, optional
        Optionally may pass a value, or list-like of values in
        which if found in either the left hemisphere or
        right hemisphere parcellation data will be set to 0.

        If passed 0 or None, will ignore.

        ::

            default = 0

    merge_thresh : float between 0 and 1, optional
         This function will handle both cases where the values
        between each file are separate (e.g., 0 to n-1 in each) or
        overlapping. The way this method distinguishes between
        these cases is by looking at the percent of labels which
        intersect between parcellation hemispheres. Specifically,
        if the number of intersecting values / the number of left hemisphere
        unique values is greater than this `merge_thresh`, the values
        in the right hemisphere data will be incremented such that they
        are treated as unique values.

        ::

            default = .75
    '''

    assert merge_thresh >= 0, 'Cannot be negative.'
    assert merge_thresh <= 1, 'Cannot be above 1.'

    # Load if needed
    lh, rh = load(lh), load(rh)

    # Process background label (set to 0 if any)
    lh = _proc_background_label(lh, background_label)
    rh = _proc_background_label(rh, background_label)
    
    # Get unique by hemisphere
    ul, ur = np.unique(lh), np.unique(rh)
    
    # Calculate intersection
    intersect = len(np.intersect1d(ul, ur))
    
    # If over 75% of labels intersect, merge
    # by adding other hemi as unique
    if intersect > 0:
        if (intersect / len(ul)) > merge_thresh:

            # Add max + 1, since parcs can start at 0
            rh[rh != 0] += (max(ul) + 1)

    # Otherwise, just concatenate as is
    data = np.concatenate([lh, rh])

    return data

def clean_parcel_labels(parcel):
    
    # Get unique regions
    u_regions = np.unique(parcel)

    # If missing 0, pretend there
    u_regions = np.union1d([0], u_regions)

    # Should be sorted, but make sure sorted
    u_regions = np.sort(u_regions)
    
    # If already 'clean' return as is
    clean_regions = np.arange(len(u_regions))
    if len(np.setdiff1d(clean_regions, u_regions)) == 0:
        return parcel
    
    # Fill in with new clean values
    clean_parcel = np.zeros(parcel.shape, dtype=parcel.dtype)
    for clean, original in zip(np.arange(len(u_regions)), u_regions):
        clean_parcel[parcel == original] = clean
        
    return clean_parcel
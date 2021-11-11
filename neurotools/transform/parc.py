import numpy as np
from ..loading import load

def _proc_background_label(data, background_label):
    '''Sets passed background label to 0'''

    # If None or 0, do nothing
    if background_label is None or background_label == 0:
        return background_label

    # If list / array-like
    if hasattr(background_label, '__iter__'):
        data[np.isin(data, background_label)] = 0

    # If non-zero single label
    else:
        data[data == background_label] = 0

    return data

def merge_parc_hemis(lh, rh, background_label=0, merge_thresh=.75):
    '''In the case that 0 should be treated as a valid label,
    pass background label == None.'''

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

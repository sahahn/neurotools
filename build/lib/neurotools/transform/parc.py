import numpy as np
from ..loading import load

def merge_parc_hemis(lh, rh, background_label=0):

    lh, rh = load(lh), load(rh)
    
    # Get unique by hemisphere
    ul, ur = np.unique(lh), np.unique(rh)
    
    # Calculate intersection
    intersect = len(np.intersect1d(ul, ur))
    
    # If over 75% of labels intersect, merge
    # by adding other hemi as unique
    if intersect > 0:
        if intersect / len(ul) > .75:

            # Add max + 1, since parcs can start at 0
            rh[rh!=background_label] += (max(ul) + 1)

    # Otherwise, just concatenate as is
    data = np.concatenate([lh, rh])

    return data

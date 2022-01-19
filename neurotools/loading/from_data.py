import os
import numpy as np

from .. import data_dr


def get_surf_loc(space, hemi, key):
    '''Find the right file based on space, hemi and key,
    with tolerance to different naming schemes'''

    # If passed list of keys, try each
    # in order, until loc is found
    if isinstance(key, list):
        for k in key:
            loc = get_surf_loc(space, hemi, key=k)
            if loc is not None:
                return loc

    # Try checking each sub
    for sub in ['surf', 'label']:
        loc = _get_surf_loc(space, hemi, key, sub)
        if loc is not None:
            return loc

    # If not found, return None for not found
    return None

def _get_surf_loc(space, hemi, key, sub):

    # Get base directory in data dr
    dr = os.path.join(data_dr, space, sub)

    # If no dr, return right away
    if not os.path.exists(dr):
        return None

    # Order determined by chance of false positive and frequency
    lh_keys = ['lh.', 'hemi-L_', 'left', 'hemi-L', 'lh',  'L_', 'L']
    rh_keys = ['rh.', 'hemi-R_', 'right', 'hemi-R', 'rh', 'R_', 'R']

    if hemi in lh_keys:
        hemi_keys = lh_keys
    elif hemi in rh_keys:
        hemi_keys = rh_keys

    # Get list of file in this directory
    files = os.listdir(dr)

    # If both a hemi key and the key of interest
    # return that full location
    for hemi_key in hemi_keys:
        for file in files:
            if key in file and hemi_key in file:
                return os.path.join(dr, file)
    
    # Otherwise, return None for not found
    return None
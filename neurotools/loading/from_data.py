import os

from .. import data_dr
from .funcs import load
from ..misc.text import clean_key
from ..misc.print import _get_print
from .space import get_space_options

# Order determined by chance of false positive and frequency
LH_KEYS = ['lh.', 'hemi-L_', 'left', 'hemi-L', 'lh_', 'lh-', 'lh',  'L_', 'L', 'l']
RH_KEYS = ['rh.', 'hemi-R_', 'right', 'hemi-R', 'rh_', 'rh-', 'rh', 'R_', 'R', 'r']


def get_surf_loc(space, hemi, key):
    '''Find the saved surface file based
    on space, hemi and key, with tolerance
    to different naming schemes.

    Parameters
    ----------
    space : str
        The space of the data, where space refers to a valid surface
        space, e.g., 'fsaverage'.
    
    hemi : str
        The hemisphere to find, can pass as 'lh' / 'rh' or
        some alternate formats, e.g., 'left' / 'right'.

    key : str or list of str
        The identifying key of the surface to load. If passing
        a list of keys, will try to find the correct surface in the
        order of the passed list, e.g., you should pass the most
        specific option first, then potentially more general options.

    Returns
    -------
    path : str or None
        The path of the saved requested surface file, or
        None if not found.
    '''

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

    # Get correct hemis to searchs
    if hemi in LH_KEYS:
        hemi_keys = LH_KEYS.copy()
    elif hemi in RH_KEYS:
        hemi_keys = RH_KEYS.copy()

    # Get list of file in this directory
    files = os.listdir(dr)

    # If both a hemi key and the key of interest
    # return that full location
    match_files = []
    for hemi_key in hemi_keys:
        for file in files:
            if key in file and hemi_key in file:
                match_files.append(file)

    # None
    if len(match_files) == 0:
        return None

    # If more than one valid file found, prefer the
    # one with the shortest length, as most likely
    # to match exactly the requested value, if just 1
    # will select that one
    match_file = sorted(match_files, key=len)[0]
    
    # Return full path
    return os.path.join(dr, match_file)

def _load_medial_wall(space, hemi):

    # Get location
    loc = get_surf_loc(space=space, hemi=hemi, key=['medial_mask', 'medialwall'])

    # Error if not found
    if loc is None:
        raise RuntimeError(f'No medial wall information found for space={space}, hemi={hemi}')

    # Return loaded as bool
    return load(loc, dtype='bool')

def auto_load_rois(data, space=None, hemi=None, verbose=None, _print=None):

    from ..plotting.ref import SurfRef, VolRef

    # Get _print object
    _print = _get_print(verbose=verbose, _print=_print)

    # Standerdize hemi if not None
    if hemi is not None:
        if hemi in LH_KEYS:
            hemi = 'lh'
        elif hemi in RH_KEYS:
            hemi = 'rh'
    
    # Get most likely parcel
    parc = _get_most_likely_parcel(data)
    _print(f'Detected most likely parcel from passed ROIs: {parc}', level=1)

    # Grab space options, starting w/ default or user passed
    if space is None:
        space = 'default'
    space_options = [space] + get_space_options()

    # Try each
    for space in space_options:
        
        try:

            # Run to check to see if works, w/ errors / warnings muted
            SurfRef(space=space, parc=parc, verbose=-1).get_hemis_plot_vals(data)

            # Not the most eff. but now we re-run again, but passing the current verbose
            # state, now that we know this space works e.g., we want to avoid
            # printing verbose for all the ones that fail.
            # Note: Passing print will override default verbose.

            sr = SurfRef(space=space, parc=parc, _print=_print)
            vals = sr.get_hemis_plot_vals(data)

            # Apply hemi if not None
            if hemi is not None:
                vals = hemi[vals]

            _print(f'Extracted plot values for space: {space}', level=1)

            return vals, space
        except (RuntimeError, FileNotFoundError):
            pass

        try:

            # See SurfRef logic / comments for this
            VolRef(space=space, parc=parc, verbose=-1).get_plot_vals(data)

            # Now actually get
            ref = VolRef(space=space, parc=parc, _print=_print)
            plot_vals = ref.get_plot_vals(data)
            
            _print(f'Extracted plot values for space: {space}', level=1)

            return plot_vals, space
        except (RuntimeError, FileNotFoundError):
            pass

    raise RuntimeError(f'Could not find matching space for detected parc: {parc}')
        
def _get_most_likely_parcel(data):

    # Get some helpful tools from ref / funcs
    from ..plotting.ref import _load_mapping
    from ..plotting.funcs import _data_to_dict

    # Load all mappings
    mappings_dr = os.path.join(data_dr, 'mappings')
    files = [f for f in os.listdir(mappings_dr) if '.mapping.txt' in f]
    mappings = {f.replace('.mapping.txt', ''): _load_mapping(os.path.join(mappings_dr, f)) for f in files}

    # Get clean name versions of passed data
    feat_names = list(_data_to_dict(data).keys())
    clean_names = [clean_key(name) for name in feat_names]

    def get_basic_overlap(parcel, mappings):
        
        # Get set of clean refs
        clean_refs = set([clean_key(k) for k in mappings[parcel].keys()] +\
                         [clean_key(v) for v in mappings[parcel].values()])
        clean_refs = [c for c in clean_refs if len(c) > 2]
        
        # Do a simple check for each parcel to try and
        # figure which one has the most
        n_found = 0
        
        # Go through each of the actual passed ROI names
        for name in clean_names:
            
            # For each one, cycle through all of the
            # possible reference keys until either one is found
            # incr and stop, or none are found
            for ref in clean_refs:
                if ref in name:
                    n_found += 1
                    continue
                    
        return n_found
    
    cnts = {parcel: get_basic_overlap(parcel, mappings) for parcel in mappings}
    mx = max(cnts.items(), key=lambda x:x[1])
    if mx[1] == 0:
        raise RuntimeError('Could not find reference parcellation to match ROI names to.')

    return mx[0]



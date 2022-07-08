from .conv import extract_subcortical_from_cifti
import nibabel as nib
import os
import pandas as pd
import numpy as np
from ..loading.funcs import load
from ..loading.from_data import _load_medial_wall, auto_load_rois
from ..loading.space import get_space_options
from .. import data_dr
from ..misc.print import _get_print

def _resolve_space_name(space):

    # Base
    if space is None:
        return None
    if space == 'native':
        return 'native'

    # Check if already okay
    space_options = os.listdir(data_dr)
    if space in space_options:
        return space

    # If not, let's let some things slide
    if space == '41k_civet':
        return 'civet'

    # Make some replacements
    space = space.replace('Civet', 'civet')
    space = space.replace('_lr', '_LR')
    space = space.replace('_FS_', '_fs_')

    # Check again
    if space in space_options:
        return space

    raise RuntimeError('Invalid space passed!')

def _load(data):
    '''Simple wrapper around load with
    the added attribute that if a str location is passed
    we want to only if the data is multi-dimensional - try to
    load it as a Nifti1Image, i.e., with affine, otherwise
    perform normal load as array.'''
    
    # If already nifti image, keep as is
    if isinstance(data, nib.Nifti1Image):
        return data
    
    # First, perform normal load
    loaded_data = load(data)
   
    # If not str, i.e., already passed as obj
    # return as is
    if not isinstance(data, str):
        return loaded_data
    
    # If loaded data is flat array, return as is
    if len(loaded_data.shape) == 1:
        return loaded_data

    # Last case is try to load instead w/ nibabel load
    # if fails, return flat array
    try:
        return nib.load(data)
    except:
        return loaded_data

def _get_space_mapping(hemi):

    # Get options for spaces
    space_options = get_space_options()
    mapping = {}

    # For each potential space
    for space in space_options:

        # See if medial wall, skip if isn't
        try:
            medial_wall = _load_medial_wall(space, hemi=hemi)
        except RuntimeError:
            continue
        
        # Add base size first
        base_sz = len(medial_wall)

        # Check if already exists
        if base_sz in mapping:

            # In case of duplicates, favor civet, then fsaverage, by skipping
            # adding new, and keeping current
            # Note, potential TODO may need more cases in future
            if 'civet' in mapping[base_sz][0]:
                continue
            if 'fsaverage' in mapping[base_sz][0]:
                continue

        # If here, then add, replacing if already there
        mapping[base_sz] = (space, False)

        # Add no medial wall version
        no_medial_sz = int(np.sum(medial_wall))
        if no_medial_sz in mapping:
            # If this happens will break later logic
            raise RuntimeError('Should not be overlaps here.')
        mapping[no_medial_sz] = (space, True)

    return mapping

def _check_rois(data):

    # The idea here is that if data is passed
    # as a pandas Series or DataFrame, then definetly
    # ROIs
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return True

    # If dict ... check to make sure not space dict
    if isinstance(data, dict):
        bad_keys = ['lh', 'rh', 'concat', 'vol', 'sub']
        for key in bad_keys:
            if key in data:
                return False

        return True

    # Return false otherwise
    return False

def process_space(data, space=None, hemi=None,
                  verbose=1, _print=None):
    '''
    If hemi=None, auto-detect. Or can pass as
    lh or rh, as this is the only case
    where it could go either way arbitrarily.
    
    Will return packed data in a tuple if lh+rh or
    lh+rh+sub.

    - Process passed sub-cortical to make sure in nifti
    - If surface data is missing medial wall, add it here

    Space argument refers to the space of the surface
    data. 
    

    Data can be passed as
    
    1. A single data representing surf, surf+surf, surf+surf+vol or just vol
    2. A single file location w/ a data array
    3. A list / array-like of length either 2 or 3, if 2 then represents surf+surf
       if 3 then surf+surf+vol
    4. A list / array-like same as above, but with file-paths
    '''

    # Resolve user passed space
    space = _resolve_space_name(space)

    _print = _get_print(verbose, _print)
    _print('Calling function process_space on data.', level=2)
    
    # Proc hemi input
    if hemi == 'left':
        hemi = 'lh'
    if hemi == 'right':
        hemi = 'rh'
    assert hemi in ['lh', 'rh', None], 'hemi must be lh, rh or None!'

    # Check to see if the data was passed as a list of ROIs + values
    # or atleast try ... 
    is_rois = _check_rois(data)
    if is_rois:
        return auto_load_rois(data, space=space, hemi=hemi, _print=_print)
    
    # Space info- sep by hemi
    lh_space_mapping = _get_space_mapping(hemi='lh')
    rh_space_mapping = _get_space_mapping(hemi='rh')
    
    # Gen combinations of hemi sizes
    hemi_sizes = []
    for lk, rk in zip(lh_space_mapping, rh_space_mapping):
        hemi_sizes.append((lk, rk))

    # TODO add cases for when just subcortical / volumetric is passed
    # Or flattened voxel or something
    # not in cifti form, but in flattened sub-cortical voxel form
    # For now these are just from cifti...
    sub_sizes = [31870, 62053]

    # Init.
    proc_data = {}
    lh_data, rh_data, vol_data = None, None, None

    # If data is passed as str - replace with loaded
    # version here - before main loop with cases
    if isinstance(data, str):
        _print(f'Loading data from location: {data}.', level=2)
        data = _load(data)

    elif isinstance(data, nib.Cifti2Image):
        # @TODO if cifti use this info directly as it contains
        # space information and whatnot. For now, just treat as array
        _print(f'Loading data from cifti as data array.', level=2)
        data = _load(data)

    # Gifti case directly
    elif isinstance(data, nib.GiftiImage):
        _print(f'Loading data from gifti as data array.', level=2)
        data = _load(data)
        
    # Otherwise, if data is already in dict form
    # check from there
    if isinstance(data, dict):

        if 'lh' in data:
            lh_data = _load(data['lh'])
        if 'rh' in data:
            rh_data = _load(data['rh'])
        if 'concat' in data:
            sz = len(data['concat']) // 2
            lh_data, rh_data = _load(data['concat'][:sz]), _load(data['concat'][sz:])

        if 'vol' in data:
            vol_data = _load(data['vol'])
        
        # Also support passing as sub
        if 'sub' in data:
            vol_data = _load(data['sub'])

    # Passed just single subcort case / volumetric
    elif isinstance(data, nib.Nifti1Image):
        vol_data = data

    # Check if
    # was passed as a list / array-like corresponding
    # to lh+rh data
    elif len(data) == 2:
        _print('Assuming data passed as [lh, rh].', level=1)
        lh_data, rh_data = _load(data[0]), _load(data[1])

    # Or lh+rh+sub case
    elif len(data) == 3:
        _print('Assuming data passed as [lh, rh, sub].', level=1)
        lh_data, rh_data, vol_data = _load(data[0]), _load(data[1]), _load(data[2])

    # Single passed array case
    # Since data is already array if here
    else:
        _print('data was passed as single array, automatically inferring splits / space.', level=2)

        for lh_sz, rh_sz in hemi_sizes:
            
            # If matches single hemi length
            if len(data) == lh_sz:
                
                # If also matches rh, and hemi is rh, override
                if len(data) == rh_sz and hemi == 'rh':
                    rh_data = data

                # Otherwise all other cases + default
                # are lh
                else:
                    lh_data = data
                break

            # Check if it matches a concat case
            elif len(data) == lh_sz + rh_sz:
                lh_data, rh_data = data[:lh_sz], data[lh_sz:]
                break

            # Go through sub cases
            for sub_sz in sub_sizes:

                # lh+rh+sub case
                if len(data) == lh_sz + rh_sz + sub_sz:
                    lh_data, rh_data = data[:lh_sz], data[lh_sz:lh_sz+rh_sz]
                    vol_data = data[lh_sz+rh_sz:]
                    break

                # Just sub case
                elif len(data) == sub_sz:
                    vol_data = data
                    break
        
        # Native space default
        if space == 'native' and lh_data is None and rh_data is None:

            if hemi is None or hemi == 'lh':
                _print('Assuming native surface space left hemisphere', level=1)
                lh_data = data
            else:
                _print('Assuming native surface space right hemisphere', level=1)
                rh_data = data

    # Make sure something found, if not error
    if lh_data is None and rh_data is None and vol_data is None:
        raise RuntimeError(f'Could not determine the space / hemi of passed data with len={len(data)}')
    
    # Process found data - don't need
    # user specified space here as still need auto-detect info
    # for if need proc medial walls.
    lh_space, rh_space = None, None
    if lh_data is not None:
        proc_data['lh'], lh_space = proc_hemi_data(lh_data, 'lh', lh_space_mapping)
    if rh_data is not None:
        proc_data['rh'], rh_space = proc_hemi_data(rh_data, 'rh', rh_space_mapping)
    if vol_data is not None:
        proc_data['vol'] = proc_vol_data(vol_data)

    _print(f'Inferred passed data: {list(proc_data)}', level=1)

    # Process space combinations
    detected_space = None
    if lh_space is None and rh_space is None:
        if 'vol' not in proc_data:
            raise RuntimeError('No valid data found')
    elif lh_space is not None and rh_space is None:
        detected_space = lh_space
    elif lh_space is None and rh_space is not None:
        detected_space = rh_space
    elif lh_space is not None and rh_space is not None:
        assert lh_space == rh_space, 'Auto detected spaces do not match!'
        detected_space = lh_space
    
    # If user passed space, use that instead
    if space is not None:
        detected_space = space
    else:
        _print(f'Inferred surface space: {detected_space}', level=1)

    # Return proc data + space
    return proc_data, detected_space

def proc_hemi_data(hemi_data, hemi, space_mapping):
    
    # If doesn't match any, assume in native space
    if len(hemi_data) not in space_mapping:
        return hemi_data, 'native'
    
    # Check space mapping dict
    space, needs_medial_wall = space_mapping[len(hemi_data)]
    
    # Add medial wall if needs it
    if needs_medial_wall:
        medial_wall_mask = _load_medial_wall(space, hemi)
        to_fill = np.zeros(medial_wall_mask.shape)

        # Medial wall saved as 0 in mask so fill in hemi data like this
        to_fill[medial_wall_mask] = hemi_data

        return to_fill, space
    
    # If already okay, return as is, with space
    return hemi_data, space

def proc_vol_data(vol_data):
    
    # If already nifti
    if isinstance(vol_data, nib.Nifti1Image):
        return vol_data

    # Otherwise, if size for cifti rois
    if len(vol_data) in [31870, 62053]:
        vol_data = extract_subcortical_from_cifti(vol_data)
    
    # The other case will be flattened subcort voxel case
    else:
        raise RuntimeError('Not implemented.')

    return vol_data

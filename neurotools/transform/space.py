from .conv import extract_subcortical_from_cifti
import nibabel as nib
import os
import numpy as np
from ..loading import load
from .. import data_dr
from ..misc.print import _get_print

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


def process_space(data, space=None, hemi=None, verbose=0, _print=None):
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
    
    1. A single data representing surf, surf+surf, surf+surf+sub or just sub
    2. A single file location w/ a data array
    3. A list / array-like of length either 2 or 3, if 2 then represents surf+surf
       if 3 then surf+surf+sub
    4. A list / array-like same as above, but with file-paths
    '''

    _print = _get_print(verbose, _print)
    _print('Calling function process_space on data.', level=2)
    
    # Proc hemi input
    if hemi == 'left':
        hemi = 'lh'
    if hemi == 'right':
        hemi = 'rh'
    assert hemi in ['lh', 'rh', None], 'hemi must be lh, rh or None!'
    
    # Space info- sep by hemi
    lh_space_mapping = {578: ('fsaverage3', True),
                        642: ('fsaverage3', False),
                        2562: ('fsaverage4', False),
                        2329: ('fsaverage4', True),
                        10242: ('fsaverage5', False),
                        9354: ('fsaverage5', True),
                        163842: ('fsaverage', False),
                        149955: ('fsaverage', True),
                        29696: ('32k_fs_LR', True),
                        32492: ('32k_fs_LR', False),
                        54216: ('59k_fs_LR', True),
                        59292: ('59k_fs_LR', False),
                        149141: ('164k_fs_LR', True)}

    rh_space_mapping = {575: ('fsaverage3', True),
                        642: ('fsaverage3', False),
                        2562: ('fsaverage4', False),
                        2332: ('fsaverage4', True),
                        10242: ('fsaverage5', False),
                        9361: ('fsaverage5', True),
                        163842: ('fsaverage', False),
                        149926: ('fsaverage', True),
                        29716: ('32k_fs_LR', True),
                        32492: ('32k_fs_LR', False),
                        54225: ('59k_fs_LR', True),
                        59292: ('59k_fs_LR', False),
                        149120: ('164k_fs_LR', True)}
    
    # Gen combinations of hemi sizes
    hemi_sizes = []
    for lk, rk in zip(lh_space_mapping, rh_space_mapping):
        hemi_sizes.append((lk, rk))

    # TODO add cases for when subcortical is passed
    # not in cifti form, but in flattened sub-cortical voxel form
    # For now these are just from cifti...
    sub_sizes = [31870, 62053]

    # Init.
    proc_data = {}
    lh_data, rh_data, sub_data = None, None, None

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
        
    # Otherwise, if data is already in dict form
    # check from there
    if isinstance(data, dict):

        if 'lh' in data:
            lh_data = _load(data['lh'])
        if 'rh' in data:
            rh_data = _load(data['rh'])
        if 'sub' in data:
            sub_data = _load(data['sub'])

    # Passed just single subcort case
    elif isinstance(data, nib.Nifti1Image):
        sub_data = data

    # Check if
    # was passed as a list / array-like corresponding
    # to lh+rh data
    elif len(data) == 2:
        _print('Assumingdata passed as [lh, rh].', level=1)
        lh_data, rh_data = _load(data[0]), _load(data[1])

    # Or lh+rh+sub case
    elif len(data) == 3:
        _print('Assuming data passed as [lh, rh, sub].', level=1)
        lh_data, rh_data, sub_data = _load(data[0]), _load(data[1]), _load(data[2])

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
                    sub_data = data[lh_sz+rh_sz:]
                    break

                # Just sub case
                elif len(data) == sub_sz:
                    sub_data = data
                    break

    # Make sure something found, if not error
    if lh_data is None and rh_data is None and sub_data is None:
        raise RuntimeError(f'Could not determine the space / hemi of passed data with len={len(data)}')
    
    # Process found data - don't need
    # user specified space here as still need auto-detect info
    # for if need proc medial walls.
    lh_space, rh_space = None, None
    if lh_data is not None:
        proc_data['lh'], lh_space = proc_hemi_data(lh_data, 'lh', lh_space_mapping)
    if rh_data is not None:
        proc_data['rh'], rh_space = proc_hemi_data(rh_data, 'rh', rh_space_mapping)
    if sub_data is not None:
        proc_data['sub'] = proc_sub_data(sub_data)

    _print(f'Inferred passed data: {list(proc_data)}', level=1)

    # Process space combinations
    detected_space = None
    if lh_space is None and rh_space is None:
        if 'sub' not in proc_data:
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

    return proc_data, detected_space

def proc_hemi_data(hemi_data, hemi, space_mapping):
    
    # If doesn't match any, assume in native space
    if len(hemi_data) not in space_mapping:
        return hemi_data, 'native'
    
    # Check space mapping dict
    space, needs_medial_wall = space_mapping[len(hemi_data)]
    
    # Add medial wall if needs it
    if needs_medial_wall:
        medial_wall_mask = np.load(os.path.join(data_dr, space, f'{hemi}_medial_mask.npy'))
        to_fill = np.zeros(medial_wall_mask.shape)

        # Medial wall saved as 0 in mask so fill in hemi data like this
        to_fill[medial_wall_mask.astype('bool')] = hemi_data

        return to_fill, space
    
    # If already okay, return as is, with space
    return hemi_data, space

def proc_sub_data(sub_data):
    
    # If already nifti
    if isinstance(sub_data, nib.Nifti1Image):
        return sub_data

    # Otherwise, if size for cifti rois
    if len(sub_data) in [31870, 62053]:
        sub_data = extract_subcortical_from_cifti(sub_data)
    
    # The other case will be flattened subcort voxel case
    else:
        raise RuntimeError('Not implemented.')

    return sub_data

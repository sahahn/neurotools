from .conv import extract_subcortical_from_cifti
import nibabel as nib
import os
import numpy as np
from .. import data_dr

def process_space(data, hemi=None):
    '''
    If hemi=None, auto-detect. Or can pass as
    lh or rh, as this is the only case
    where it could go either way arbitrarily.
    
    Will return packed data in a tuple if lh+rh or
    lh+rh+sub.

    - Process passed sub-cortical to make sure in nifti
    - If surface data is missing medial wall, add it here
    
    '''
    
    # Proc hemi input
    if hemi == 'left':
        hemi = 'lh'
    if hemi == 'right':
        hemi = 'rh'
    assert hemi in ['lh', 'rh', None], 'hemi must be lh, rh or None'
    
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

    # First case is to see if data
    # was passed as a list / array-like corresponding
    # to lh+rh data
    if len(data) == 2:
        lh_data, rh_data = data[0], data[1]

    # Or lh+rh+sub case
    elif len(data) == 3:
        lh_data, rh_data, sub_data = data[0], data[1], data[2]

    # Single passed array case
    else:

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

    space = None
    if lh_data is not None:
        proc_data['lh'], space = proc_hemi_data(lh_data, 'lh', lh_space_mapping)
    if rh_data is not None:
        proc_data['rh'], space = proc_hemi_data(rh_data, 'rh', rh_space_mapping)
    if sub_data is not None:
        proc_data['sub'] = proc_sub_data(sub_data)

    return proc_data, space

def proc_hemi_data(hemi_data, hemi, space_mapping):

    space, needs_medial_wall = space_mapping[len(hemi_data)]
    
    # Add medial wall if needs it
    if needs_medial_wall:
        medial_wall_mask = np.load(os.path.join(data_dr, space, f'{hemi}_medial_mask.npy'))
        to_fill = np.zeros(medial_wall_mask.shape)

        # Medial wall saved as 0 in mask so fill in hemi data like this
        to_fill[medial_wall_mask] = hemi_data

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

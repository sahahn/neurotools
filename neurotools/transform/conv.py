from re import sub
import nibabel as nib
import numpy as np
import pickle as pkl
import os
from nibabel.cifti2 import Cifti2BrainModel
from .. import data_dr
from ..loading import load


def remove_medial_wall(fill_cifti, parcel, index_map):
    '''Remove the medial wall from a parcel / surface. For
    now assumes that the passed index map is for a cifti file with medial
    wall included.'''

    for index in index_map:
        if isinstance(index, Cifti2BrainModel):
            if index.surface_number_of_vertices is not None:
                inds = index.index_offset + np.array(index.vertex_indices._indices)
                fill_cifti[index.index_offset: index.index_offset+index.index_count] = parcel[inds]

    return fill_cifti

def add_subcortical(fill_cifti, index_map):
    '''Assumes base parcel already added'''

    # Start at 1 plus last
    cnt = np.max(np.unique(fill_cifti)) + 1

    for index in index_map:
        if isinstance(index, Cifti2BrainModel):
            
            # If subcort volume
            if index.surface_number_of_vertices is None:
                print(index.brain_structure)
                start = index.index_offset 
                end = start + index.index_count
                fill_cifti[start:end] = cnt
                cnt += 1

    return fill_cifti

def static_parc_to_cifti(parcel, index_map, add_sub=True):

    # Init empty cifti with zeros
    if add_sub:
        fill_cifti = np.zeros(91282)
    else:
        fill_cifti = np.zeros(59412)

    # Apply cifti medial wall reduction to parcellation
    fill_cifti = remove_medial_wall(fill_cifti, parcel, index_map)
    
    # Add subcortical structures as unique parcels next
    if add_sub:
        fill_cifti = add_subcortical(fill_cifti, index_map)

    return fill_cifti


def get_cort_slabs(parcel, index_map):
    '''Prob. case'''
    
    cort_slabs = []
    for i in range(parcel.shape[1]):
        slab = np.zeros(91282)
        slab = remove_medial_wall(slab, parcel[:, i], index_map)
        cort_slabs.append(slab)
        
    return cort_slabs
        

def get_subcort_slabs(index_map):
    'Prob. case'
    
    subcort_slabs = []

    for index in index_map:
        if isinstance(index, Cifti2BrainModel):
            if index.surface_number_of_vertices is None:

                slab = np.zeros(91282)
                
                start = index.index_offset 
                end = start + index.index_count
                slab[start:end] = 1
                
                subcort_slabs.append(slab)
                
    return subcort_slabs


def prob_parc_to_cifti(parcel, index_map):
    'Prob. case'

    cort_slabs = get_cort_slabs(parcel, index_map)
    subcort_slabs = get_subcort_slabs(index_map)
    
    return np.stack(cort_slabs + subcort_slabs, axis=1)


def surf_parc_to_cifti(cifti_file, parcel, add_sub=True):
    '''For now just works when parcel file is a parcellation
    in combined fs_LR_32k lh+rh space with medial wall included.
    Works for static or prob.'''

    # Get index map from example cifti file
    index_map = nib.load(cifti_file).header.get_index_map(1)

    # Load parcel
    parcel = load(parcel)
    
    # Probabilistic case
    if len(parcel.shape) > 1:
        return prob_parc_to_cifti(parcel, index_map)

    # Static case
    return static_parc_to_cifti(parcel, index_map, add_sub=add_sub)

def add_surface_medial_walls(data):
    '''Right now only works with just concat'ed surface level data'''
    
    # Load index maps
    if isinstance(data, nib.Cifti2Image):
        index_map = data.header.get_index_map(1)
    else:
        index_map = nib.cifti2.load(data).header.get_index_map(1)

    lh_index, rh_index = index_map[0], index_map[1]
    
    # Make sure right type of cifti, i.e., with medial walls missing
    assert lh_index.surface_number_of_vertices != lh_index.index_count
    
    # Init to fill
    n_vert_per_hemi = lh_index.surface_number_of_vertices
    to_fill = np.zeros(int(n_vert_per_hemi * 2))
    
    # Get separate inds
    lh_inds = np.array(lh_index.vertex_indices._indices)
    rh_inds = np.array(rh_index.vertex_indices._indices)
    
    # Load also as data - okay if already cifti image
    np_data = load(data)
    
    # Fill in
    to_fill[lh_inds] = np_data[:len(lh_inds)]
    to_fill[rh_inds + n_vert_per_hemi] = np_data[len(lh_inds):]
    
    return to_fill

def _get_index_map(data):

    if len(data.shape) == 1:
        cifti_len = data.shape[0]
    elif len(data.shape) == 2:
        cifti_len = data.shape[1]
    else:
        raise RuntimeError('Passed data cannot exceed 2-dimensions!')

    # Hacky solution for now - for when subcort already
    # extracted.
    sub_cifti_mapping = {31870: 91282, 62053: 170494}

    if cifti_len in sub_cifti_mapping:
        cifti_len = sub_cifti_mapping[cifti_len]

    # Get loc if saved
    index_map_loc = os.path.join(data_dr, 'index_maps',
                                 str(int(cifti_len)) + '.pkl')

    if not os.path.exists(index_map_loc):
        raise RuntimeError(f'Cannot find saved cifti index map for cifti length = {cifti_len}!')

    with open(index_map_loc, 'rb') as f:
        index_map = pkl.load(f)

    return index_map

def extract_subcortical_from_cifti(data, index_map=None, space=None):
    '''Given some cifti-data like array and an index_map, 
    extract just the sub-cortical and return as a Nifti1Image
    
    If data is multi-dimensional, assume that the 2nd part of the shape
    corresponds to the cifti-data.
    '''

    from ..plotting.ref import VolRef, _get_vol_ref_from_guess

    # Make sure data has no hanging dimensions
    data = np.squeeze(data)
    
    # Get index map from shape of data if not passed
    if index_map is None:
        index_map = _get_index_map(data)

    # Go through each index map
    sub_mapping, offsets = [], []
    
    for index in index_map:
        if isinstance(index, Cifti2BrainModel):
            if index.surface_number_of_vertices is None:

                # Add ijk mapping
                sub_mapping += index.voxel_indices_ijk._indices

                # Add offsets
                offsets.append(index.index_offset)
    
    # Concat mapping
    sub_mapping = tuple(np.stack(sub_mapping, axis=1))
                
    # Assumes that subcortical are saved at end of cifti array
    index_offset = min(offsets)
    
    # Get vol ref from guess, or if passed
    if space is None:
        vol_ref = _get_vol_ref_from_guess(sub_mapping)
    else:
        vol_ref = VolRef(space=space)
    
    # Fill empty with data based on extracted mapping
    to_fill = np.zeros(vol_ref.shape)
    
    # If doesn't match then try case where no index offset
    # since subcortical already is extracted.
    if len(to_fill[sub_mapping]) != len(data[index_offset:]):
        index_offset = 0

    if len(data.shape) == 1:
        to_fill[sub_mapping] = data[index_offset:]
    elif len(data.shape) == 2:
        to_fill[sub_mapping] = data[:, index_offset:]

    else:
        raise RuntimeError('Passed data cannot exceed 2-dimensions!')

    # Return as Nifti1Image
    return nib.Nifti1Image(to_fill, vol_ref.ref_vol_affine)



'''
Desired future functionality.

- Should be able to add / remove medial wall.
    - Input should be cifti, gifti, or array (i.e., need to have saved data references)
    - Input should be either just lh, just rh, concat surfaces, or full cifti w/ subcortical

- Parcellations should be able to be converted from lh + rh separate or concat,
  to full cifti, note that this isn't possible for normal surfaces. Functionality
  already exists, except should handle some extra cases.

- Would be cool to be able to provide re-sampling between different surfaces

'''
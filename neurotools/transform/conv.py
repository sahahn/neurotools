import nibabel as nib
import numpy as np
from nibabel.cifti2 import Cifti2BrainModel
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
                
                start = index.index_offset 
                end = start + index.index_count
                fill_cifti[start:end] = cnt
                cnt += 1

    return fill_cifti

def static_parc_to_cifti(parcel, index_map, add_subcortical=True):

    # Init empty cifti with zeros
    if add_subcortical:
        fill_cifti = np.zeros(91282)
    else:
        fill_cifti = np.zeros(59412)

    # Apply cifti medial wall reduction to parcellation
    fill_cifti = remove_medial_wall(fill_cifti, parcel, index_map)
    
    # Add subcortical structures as unique parcels next
    if add_subcortical:
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


def surf_parc_to_cifti(cifti_file, parcel_file):
    '''For now just works when parcel file is a parcellation
    in combined fs_LR_32k lh+rh space with medial wall included.
    Works for static or prob.'''

    # Get index map from example cifti file
    index_map = nib.load(cifti_file).header.get_index_map(1)

    # Load parcel
    parcel = load(parcel_file)
    
    # Probabilistic case
    if len(parcel.shape) > 1:
        return prob_parc_to_cifti(parcel, index_map)

    # Static case
    return static_parc_to_cifti(parcel, index_map)

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
    
    # Get seperate inds
    lh_inds = np.array(lh_index.vertex_indices._indices)
    rh_inds = np.array(rh_index.vertex_indices._indices)
    
    # Load also as data - okay if already cifti image
    np_data = load(data)
    
    # Fill in
    to_fill[lh_inds] = np_data[:len(lh_inds)]
    to_fill[rh_inds + n_vert_per_hemi] = np_data[len(lh_inds):]
    
    return to_fill
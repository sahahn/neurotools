from nibabel.freesurfer.io import read_annot, read_geometry
import numpy as np
import nibabel as nib
import pandas as pd
import os
from ..misc.text import get_unique_str_markers
from ..misc.print import _get_print
from .. import data_dr as def_data_dr
from ..loading.from_data import get_surf_loc
from difflib import SequenceMatcher
from ..misc.text import clean_key
from itertools import permutations
from .funcs import _data_to_dict

import warnings
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    from nilearn.surface import load_surf_data

# Custom exception
class RepeatROIError(Exception):
    pass

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def _save_mapping(mapping, loc):
    
    with open(loc, 'w') as f:
        for m in mapping:
            f.write(str(m))
            f.write(',')
            f.write(str(mapping[m]))
            f.write('\n')

def _load_mapping(loc):

    mapping = {}

    with open(loc, 'r') as f:
        for line in f.readlines():
            line = line.split(',')
            mapping[line[0]] = line[1].rstrip()

    return mapping



def auto_determine_lh_rh_keys(data):

    # Use static key patterns to search for
    from ..loading.from_data import LH_KEYS, RH_KEYS
    
    # Get feat names as list from data
    feat_names = list(_data_to_dict(data).keys())

    # Compute tolerance
    tol = .15
    tol_n = len(feat_names) * tol
    tol_upper = len(feat_names) // 2 + tol_n
    tol_lower = len(feat_names) // 2 - tol_n

    # Search through each option
    for lh, rh in zip(LH_KEYS, RH_KEYS):

        # Check startswith, then endswith, then just is in
        for key_type in ['start', 'end', None]: 

            # Count number in each, depending on key type
            if key_type is None:
                n_lh = len([f for f in feat_names if lh in f])
                n_rh = len([f for f in feat_names if rh in f])
            elif key_type == 'start':
                n_lh = len([f for f in feat_names if f.startswith(lh)])
                n_rh = len([f for f in feat_names if f.startswith(rh)])
            elif key_type == 'end':
                n_lh = len([f for f in feat_names if f.endswith(lh)])
                n_rh = len([f for f in feat_names if f.endswith(rh)])
        
            # Should be around same size, but 
            # doesn't have to be exactly equal if
            # extending to other ROIs in future
            if n_lh > tol_lower and n_lh < tol_upper and n_rh > tol_lower and n_rh < tol_upper:
                return (lh, key_type), (rh, key_type)

    raise RuntimeError('Could not auto determine lh and rh unique keys.')

def _is_in(name, key):

    # Unpack tuple key
    k, k_type = key

    # Base is in
    if k_type is None:
        return k in name
    elif k_type == 'start':
        return name.startswith(k)
    elif k_type == 'end':
        return name.endswith(k)
    else:
        raise RuntimeError('Invalid key type.')

def _replace(name, key, new):

      # Unpack tuple key
    k, k_type = key

    if k_type is None:
        return name.replace(k, new)
    elif k_type == 'start':
        if name.startswith(k):
            return name[len(k):]
        return name
    elif k_type == 'end':
        if name.endswith(k):
            return name[:len(k)]
        return name
    else:
        raise RuntimeError('Invalid key type.')

def _get_pieces(key):
    return set([k for k in key.split(' ') if len(k) > 0])

def _get_roi_dict(data, i_keys=None, d_keys=None):
    '''If data is a df, with assume that the data is stored in two columns
    with name of ROI in one, and value in the other. Otherwise, if data is 
    a dictionary, assume that it is already in name of ROI, value form.

    i_keys is a list of keys where all of the entries if any have to present
    in the name of the roi for it to be kept, d_keys is a list / single item
    where if any of the d_keys are present in an roi it will be dropped.
    '''

    # Get as dict
    data = _data_to_dict(data)

    # Make a dict processing the passed i and d keys
    spec_data = {}
    for name in data:

        # In order to add, must qualify for all i_keys, and not any d_keys
        if all([_is_in(name, key) for key in i_keys]) and not any([_is_in(name, key) for key in d_keys]):
            spec_data[name] = data[name]

    # Generate list of alt names
    try:
        u_markers = get_unique_str_markers(list(spec_data))

    # If fails, then pass same as base
    except IndexError:
        u_markers = list(spec_data)

    return spec_data, u_markers

def _check_combos(not_found, name_pieces):
    
    # If size 0 or size 1, then return
    # as combo search is over
    if len(not_found) <= 1:
        return not_found
    
    # Generate all possible combinations, checking sequentially
    for n in range(2, len(not_found)+1):
        ps = permutations(not_found, n)
        
        # Check each combo
        for p in ps:
            combo = ''.join(p)
            
            # If present, call recursively
            # but with each piece removed from
            # not found
            if combo in name_pieces:
                for name in p:
                    not_found.remove(name)
                    
                return _check_combos(not_found, name_pieces)

    # If no matches, return as is
    return not_found

class Ref():
    
    def __init__(self, space, parc, data_dr='default', verbose=0, _print=None):
               
        self.space = space
        self.parc = parc

        # Get verbose print object
        self._print = _get_print(verbose, _print=_print)

        if data_dr == 'default':
            data_dr = def_data_dr

        self.data_dr = data_dr
        
        self._load_mappings()
        self._load_ref()
        
    def _load_mappings(self):

        if self.parc is None:
            self.mapping, self.label_2_int = None, None
            return

        map_loc = os.path.join(self.data_dr, 'mappings', self.parc + '.')

        try:
            self.mapping = _load_mapping(map_loc + 'mapping.txt')
            self.label_2_int = _load_mapping(map_loc + 'label_2_int.txt')
            self._print(f'Loaded mapping and label_2_int dicts, from: {map_loc}', level=2)
        except FileNotFoundError:
            self.mapping, self.label_2_int = None, None
    
    def _load_ref(self):
        pass
    
    def _get_ref_vals(self, hemi=None):
        pass

    def _add_to_inds(self, inds, name, label, trans_label):

        sim = similar(trans_label, name)
        ind = int(self.label_2_int[label])
        inds.append((ind, label, sim))
        self._print(f'Found match {trans_label} w/ similarity={sim}', level=3)

    def _proc_keys_input(self, keys):

        if keys is None:
            keys = []

        elif isinstance(keys, str):
            keys = [keys]

        elif isinstance(keys, tuple):
            if len(keys) != 2:
                raise RuntimeError(f'Invalid passed for keys={keys}')
            keys = [keys]

        elif isinstance(keys, list):
            for i, key in enumerate(keys):

                if isinstance(key, tuple) and len(key) == 2:
                    if key[1] not in [None, 'start', 'end']:
                        raise RuntimeError(f'Passed {key} if tuple, 2nd element must be None, start or end.')
                
                elif not isinstance(key, str):
                    raise RuntimeError(f'Passed element in keys {key} is not a valid str.')
                
                # If here, than the key is a single str, wrap in tuple w/ None
                else:
                    keys[i] = (key, None)
        
        else:
            raise RuntimeError(f'keys must be str or list of str / tuple.')

        return keys

    def _check_for_pieces_ind(self, name, check_combos=True):

        # Alternate case if sub-string not found
        # designed to get at cases where the order might be wrong
        inds = []
        
        # Get names as pieces representation
        name_pieces = _get_pieces(name)

        for label in self.label_2_int:

            # Use the cleaned version of each
            trans_label = clean_key(label)
            trans_pieces = _get_pieces(trans_label)

            # Compute intersection
            piece_intersection = trans_pieces.intersection(name_pieces)

            # If the same
            if len(piece_intersection) == len(trans_pieces):
                self._add_to_inds(inds, name, label, trans_label)

            # Otherwise, check recursively for combinations
            elif check_combos:

                not_found = trans_pieces - piece_intersection
                not_found = _check_combos(not_found, name_pieces)
                
                # If empty, found
                if len(not_found) == 0:
                    self._add_to_inds(inds, name, label, trans_label)

        return inds

    def _check_for_ind(self, name, original_name):

        # Try to find the correct index
        inds = []

        # These labels are the ground truth we need
        # to map to the saved index
        for label in self.label_2_int:

            # Use the cleaned version of each
            trans_label = clean_key(label)

            # If the ground truth is sub-string of the current name
            # Keep track, w/ also a simmilarity score
            if trans_label in name:
                ind = int(self.label_2_int[label])

                # Append ind, label and simmilarity
                self._add_to_inds(inds, name, label, trans_label)

        # If none found from base index search, use pieces method
        if len(inds) == 0:
            inds  = self._check_for_pieces_ind(name)

        # If more than one, sort so first is highest simmilarity
        if len(inds) > 1:
            inds = sorted(inds, key=lambda i : i[2], reverse=True)

        #  Return first, if only one then that is first
        if len(inds) > 0:
            self._print(f'Mapping: {original_name} -> {inds[0][1]}.', level=2)
            return inds[0][0]

        # If not found, return None
        return None

    def _apply_mappings_to_name(self, name):

        self._print(f'Applying mapping to {name}', level=2)

        # Apply the mapping for this parcellation
        for key in self.mapping:

            # Get a clean representation of a possible transform key
            trans_key = clean_key(key)

            # If found in the current ROI name, replace it
            if trans_key in name:

                # Get the cleaned new value to replace with
                new = clean_key(self.mapping[key])
                name = name.replace(trans_key, new)
                self._print(f'replace: "{trans_key}":"{new}", new="{name}"', level=3)

        return name
        

    def _find_ref_ind(self, name, alt_name=None, i_keys=None):

        # Keep copy of original name for verbose print
        original_name = name
        
        # Base clean name + store copy
        name = clean_key(name)
        clean_name = name
        self._print(f'clean_key: "{original_name}" to "{name}"', level=3)

        # Try applying mappings to name, then perform first check 
        name = self._apply_mappings_to_name(name)
        ind = self._check_for_ind(name, original_name)
        if ind is not None:
            return ind

        # Next, let's try with the pre-mapped name
        ind = self._check_for_ind(clean_name, original_name)
        if ind is not None:
            return ind

        # Next, try again but with all i_keys removed from the name
        if i_keys is not None:
            next_name = original_name
            for key in i_keys:
                next_name = _replace(next_name, key, '')

            self._print(f'Did not find roi for "{original_name}", trying with "{next_name}"', level=2)
            
            # We make sure to pass i_keys as None here, to avoid infinite loops
            ind = self._find_ref_ind(next_name, alt_name=alt_name, i_keys=None)
            if ind is not None:
                return ind

        # If still haven't found anything, try everything again but with alternate name
        if alt_name is not None:
            self._print(f'Did not find roi for "{name}", trying with "{alt_name}"', level=2)

            # We make sure to pass alt_name as None here, to avoid infinite loops
            ind = self._find_ref_ind(alt_name, alt_name=None, i_keys=i_keys)
            if ind is not None:
                return ind

        # If finally not found, return None
        return None

    def find_ref_ind(self, name, alt_name=None, i_keys=None):

        # Try to find ind
        ind = self._find_ref_ind(name=name, alt_name=alt_name, i_keys=i_keys)
        if ind is not None:
            return ind

         # If error, print out the true region names
        self._print('The passed values must be able to map in some way to one of these reference region names:',
                    list(self.label_2_int), 'You can turn on verbose to see what is already being checked.', level=0)

        # If didn't find with alt name also
        raise RuntimeError(f'Could not find matching ROI for: "{name}"')
    
    def _get_plot_vals(self, roi_dict,
                       roi_alt_names,
                       ref_vals, i_keys, use_just_alt=False):

         # Init plot vals
        plot_vals = np.zeros(np.shape(ref_vals))

        # Keep track of already found inds
        used_inds = set()
        for name, alt_name in zip(roi_dict, roi_alt_names):

            # Get plot value
            value = roi_dict[name]

            # Optionally, override name w/ alt_name
            if use_just_alt:
                name = alt_name

            # Try to find the name in the reference values
            ind = self.find_ref_ind(name=name, alt_name=alt_name, i_keys=i_keys)

            # Check to see if this ROI is mapping to 
            # one another has already mapped to,
            # if so, trigger an error.
            if ind in used_inds:
                self._print('Overlapping ROI mapping error.', level=2)
                raise RepeatROIError(f'Error: mapping {name} failed, as another ROI already mapped to the same reference (ind={ind}). '\
                                      'Set verbose>=2 in Ref object and double check how each ROI is being mapped to make sure it is correct. ')
            else:
                used_inds.add(ind)

            # Set the proper values based on the found index
            plot_vals = np.where(ref_vals == ind, value, plot_vals)

        return plot_vals

    def get_plot_vals(self, data, hemi=None, i_keys=None, d_keys=None):

        self._print(f'Start get plot vals for hemi={hemi}', level=1)

        # Process keys input
        i_keys, d_keys = self._proc_keys_input(i_keys), self._proc_keys_input(d_keys)
        
        # Get base roi dict
        roi_dict, roi_alt_names = _get_roi_dict(data, i_keys, d_keys)
        self._print(f'roi_alt_names={roi_alt_names}', level=3)
        
        # Load reference values
        ref_vals = self._get_ref_vals(hemi)

        # Try - but in case of repeat ROI error
        # then try again, but with use_just_alt
        try:
            plot_vals = self._get_plot_vals(roi_dict, roi_alt_names,
                                            ref_vals, i_keys, use_just_alt=False)
        except RepeatROIError:
            self._print(f'Trying plot vals w/ use_just_alt=True', level=2)
            plot_vals = self._get_plot_vals(roi_dict, roi_alt_names,
                                            ref_vals, i_keys, use_just_alt=True)
       
        return plot_vals

class SurfRef(Ref):
    
    def __init__(self, space='fsaverage5', parc='destr',
                 data_dr='default', surf_mesh=None,
                 bg_map=None, verbose=0, _print=None):

        if space == 'default':
            space = 'fsaverage5'

        super().__init__(space, parc, data_dr, verbose=verbose, _print=_print)

        self.surf_mesh = surf_mesh
        self.bg_map = bg_map
    
    def _load_ref(self):
        
        # If no parc, skip
        if self.parc is None:
            return

        ref_loc = os.path.join(self.data_dr, self.space, 'label')
        self._print(f'Using base ref_loc = {ref_loc}', level=2)

        # Try to load lh
        lh_loc = os.path.join(ref_loc, 'lh.' + self.parc)

        if os.path.exists(lh_loc + '.annot'):
            self.lh_ref = read_annot(lh_loc + '.annot')[0]
        elif os.path.exists(lh_loc + '.gii'):
            self.lh_ref = load_surf_data(lh_loc + '.gii')
        elif os.path.exists(lh_loc + '.npy'):
            self.lh_ref = np.load(lh_loc + '.npy')
        else:
            raise RuntimeError('Could not find lh ref parcel for this space')

        # Try to load rh
        rh_loc = os.path.join(ref_loc, 'rh.' + self.parc)
        if os.path.exists(rh_loc + '.annot'):
            self.rh_ref = read_annot(rh_loc + '.annot')[0]
        elif os.path.exists(rh_loc + '.gii'):
            self.rh_ref = load_surf_data(rh_loc + '.gii')
        elif os.path.exists(rh_loc + '.npy'):
            self.rh_ref = np.load(rh_loc + '.npy')
        else:
            raise RuntimeError('Could not find rh ref parcel for this space')

    def _get_ref_vals(self, hemi):
        
        if hemi == 'lh' or hemi == 'left':
            ref_vals = self.lh_ref
        else:
            ref_vals = self.rh_ref
    
        return ref_vals
    
    def get_hemis_plot_vals(self, data, lh_key='auto', rh_key='auto', i_keys=None, d_keys=None):

        # Check  to see if either lh_key or rh_key is passed as auto
        if lh_key == 'auto' or rh_key == 'auto':
            
            # Get auto determined
            lh_a_key, rh_a_key = auto_determine_lh_rh_keys(data)
            self._print(f'Auto determined keys as lh_key={lh_a_key}, rh_key={rh_a_key}.', level=1)

            # Only override passed value for one if auto
            if lh_key == 'auto':
                lh_key = lh_a_key
            if rh_key == 'auto':
                rh_key = rh_a_key

        # Process input
        i_keys, d_keys = self._proc_keys_input(i_keys), self._proc_keys_input(d_keys)
        
        # Get the plot values per hemisphere seperately
        lh_plot_vals = self.get_plot_vals(data, 'lh', i_keys+[lh_key], d_keys)
        rh_plot_vals = self.get_plot_vals(data, 'rh', i_keys+[rh_key], d_keys)
        
        return {'lh': lh_plot_vals, 'rh': rh_plot_vals}

    def get_surf(self, name, hemi):

        if name is None:
            return None
        
        # If already surf mesh like
        if not isinstance(name, str):
            return name
        
        # Assume name is path if exists
        if os.path.exists(name):
            loc = name

        # Otherwise find right loc if any
        else:
            loc = get_surf_loc(space=self.space, hemi=hemi, key=name)
            if loc is None:
                raise  RuntimeError(f'Unable to find space={self.space}, hemi={hemi}, name={name}')

        # If gifti
        if loc.endswith('.gii'):
            surf = load_surf_data(loc)
            if len(surf) == 2:
                surf = (surf[0], surf[1])
            return surf

        # Otherwise try geometry first then base surf data
        try:
            return read_geometry(loc)
        except ValueError:
            return load_surf_data(loc)

class VolRef(Ref):
    
    def __init__(self, space='mni_1mm', parc='aseg', data_dr='default',
                 verbose=0, _print=None):

        if space == 'default':
            space = 'mni_1mm'

        super().__init__(space, parc, data_dr, verbose=verbose, _print=_print)
    
    @property
    def shape(self):
        return self.ref_vol.shape
    
    def _load_ref(self):
        
        ref_loc = os.path.join(self.data_dr, self.space)
        self._print(f'Using base ref_loc = {ref_loc}', level=2)
        
        # Get w/ flexible to different file extension
        options = os.listdir(ref_loc)
        options_no_ext = [o.split('.')[0] for o in options]

        try:
            ind = options_no_ext.index(self.parc)

        # If doesn't exist, help user
        except ValueError:
            self._print(f'Note valid parc options for {self.space} are:', options_no_ext, level=0)
            raise RuntimeError(f'Space: {self.space} does not have parc {self.parc}.')

        # Load with nibabel, since need affine
        ref_vol_raw = nib.load(os.path.join(ref_loc, options[ind]))

        # Save to class
        self.ref_vol_affine = ref_vol_raw.affine
        self.ref_vol = np.array(ref_vol_raw.get_fdata())
        
    def _get_ref_vals(self, hemi=None):
        return self.ref_vol
    
    def get_plot_vals(self, data, hemi=None, i_keys=[], d_keys=[]):
        
        plot_vals = super().get_plot_vals(data, hemi, i_keys, d_keys)
        return nib.Nifti1Image(plot_vals, self.ref_vol_affine)


def _get_vol_ref_from_guess(voxel_inds):

    # Just these options for now - any others should add? TODO
    # Note: Favors hcp_rois parc since those are cifti native

    # MNI 2mm
    if np.max(voxel_inds) == 76:
        return VolRef(space='mni_2mm', parc='hcp_rois')
   
    # MNI 1.6mm
    elif np.max(voxel_inds) == 95:
        return VolRef(space='mni_1.6mm', parc='hcp_rois')

    # MNI 1mm as base case
    return VolRef(space='mni_1mm', parc='aseg')

    
    







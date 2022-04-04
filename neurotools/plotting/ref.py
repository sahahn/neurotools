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

import warnings
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    from nilearn.surface import load_surf_data

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

def _get_col_names(df):
    
    all_cols = list(df)

    for a in all_cols:
        try:
            df[a].astype('float')
            value_col = a
        except ValueError:
            name_col = a

    return name_col, value_col

def _data_to_dict(data):

    # If series convert to dict directly
    if isinstance(data, pd.Series):
        return data.to_dict()
        
    # If df
    elif isinstance(data, pd.DataFrame):

        # If only 1 col, use index as name col
        if len(list(data)) == 1:
            return data[list(data)[0]].to_dict()

        # If 2 col DF proc
        as_dict = {}
        name_col, value_col = _get_col_names(data)

        for i in data.index:
            name = data[name_col].loc[i]
            as_dict[name] = data[value_col].loc[i]

        return as_dict

    elif isinstance(data, dict):
        return data

    raise RuntimeError('Invalid format passed for data, must be dict, pd.Series or pd.DataFrame')

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
    u_markers = get_unique_str_markers(list(spec_data))

    return spec_data, u_markers

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

    def _find_ref_ind(self, name, alt_name=None, i_keys=None):

        # Keep copy of original name for verbose print
        original_name = name

        # Base transform roi name
        name = clean_key(name)

        # Apply the mapping
        for key in self.mapping:
            trans_key = clean_key(key)

            if trans_key in name:
                name = name.replace(trans_key, clean_key(self.mapping[key]))

        # Find the ind
        inds = []
        for label in self.label_2_int:
            trans_label = clean_key(label)

            # Try to find name, if found keep track
            if trans_label in name:
                ind = int(self.label_2_int[label])

                # Append ind, label and simmilarity
                inds.append((ind, label, similar(trans_label, name)))

        # If more than one, sort so first is highest simmilarity
        if len(inds) > 1:
            inds = sorted(inds, key=lambda i : i[2], reverse=True)

        #  Return first
        if len(inds) > 0:
            self._print(f'Mapping: {original_name} -> {inds[0][1]}.', level=2)
            return inds[0][0]

        # First pass if doesn't find is to try again, but with all i_keys removed
        if i_keys is not None:
            next_name = original_name
            for key in i_keys:
                next_name = next_name.replace(key, '')

            self._print(f'Did not find roi for {original_name}, trying with {next_name}', level=2)
            return self._find_ref_ind(next_name, alt_name=alt_name, i_keys=None)

        # If still didn't find, try again with the passed alt name
        if alt_name is not None:
            self._print(f'Did not find roi for {name}, trying with {alt_name}', level=2)
            return self._find_ref_ind(alt_name, alt_name=None, i_keys=i_keys)

        # If error, print out the true region names
        self._print('Note that the passed values must be able to map on in some way to one of these regions:',
                    list(self.label_2_int), level=0)

        # If didn't find with alt name also
        raise RuntimeError(f'Could not find matching roi for {name}!')
        
    
    def get_plot_vals(self, data, hemi=None, i_keys=None, d_keys=None):

        self._print(f'Start get plot vals for hemi={hemi}', level=1)

        # Process keys input
        i_keys, d_keys = self._proc_keys_input(i_keys), self._proc_keys_input(d_keys)
        
        # Get base roi dict
        roi_dict, roi_alt_names = _get_roi_dict(data, i_keys, d_keys)
        
        # Get ref vals
        ref_vals = self._get_ref_vals(hemi)

        # Init plot vals
        plot_vals = np.zeros(np.shape(ref_vals))

        # Keep track of already found inds
        used_inds = set()
        for name, alt_name in zip(roi_dict, roi_alt_names):

            # Get plot value
            value = roi_dict[name]

            # Try to find the name in the reference values
            ind = self._find_ref_ind(name=name, alt_name=alt_name, i_keys=i_keys)

            # Check to see if this ROI is mapping to 
            # one another has already mapped to,
            # if so, trigger an error.
            if ind in used_inds:
                raise RuntimeError(f'Error: mapping {name} failed, as another ROI already mapped to the same referece (ind={ind}). '\
                                   'Set verbose>=2 in Ref object and double check how each ROI is being mapped to make sure it is correct. ')
            else:
                used_inds.add(ind)

            # Set the proper values based on the found index
            plot_vals = np.where(ref_vals == ind, value, plot_vals)

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

    
    







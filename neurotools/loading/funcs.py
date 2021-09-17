import nibabel as nib
import numpy as np
import os
import time

from joblib import Parallel, delayed
from nibabel import freesurfer as fs
from nibabel import GiftiImage

def _get_print(verbose, _print=None):
    
    # If already set
    if not _print is None:
        return _print

    def _print(*args, **kwargs):

        if 'level' in kwargs:
            level = kwargs.pop('level')
        else:
            level = 1

        if verbose >= level:
            print(*args, **kwargs, flush=True)

    return _print

def load(f, index_slice=None):
    '''Smart loading function for neuro-imaging type data.

    Parameters
    ----------
    f : file path
        The location / file path of the data to load.

    index_slice : slices, tuple of slices, or None
        You may optional pass index slicing here.
        The typical benefit of index_slicing over masking in a later
        step is where data to load will be read with nibabel
        (e.g., saved as .nii or .mgz) and can therefore use an index_slice
        to load in only the data of interest. 

        For example let's say the shape of data is 5 x 20 on
        the disk, but you only need the data in the first 20, e.g.,
        so with traditional slicing that is [0], in this case by using the
        mask you have to load the full data from the disk, then slice it.
        But if you pass `slice(0)` here, then this accomplishes the same thing,
        but only loads the requested slice from the disk.

        You must use python keyword `slice` for complex slicing. For example
        in the example above you may pass either `(0)` or `slice(0)`,
        but if you wanted something more complex, e.g. passing
        something like my_array[1:5:2, ::3] you should pass
        (slice(1,5,2), slice(None,None,3)) here.

        ::

            default = None
    '''
    
    # If already numpy array, return of as is
    if type(f) is np.ndarray:
        raw = f.copy()

    # Keep as None if None
    elif f is None:
        return None
    
    # If loading from file
    elif isinstance(f, str):
    
        # Freesurfer cases (from nilearn load func)
        if (f.endswith('area')
            or f.endswith('curv')
            or f.endswith('sulc')
            or f.endswith('thickness')):
            raw = fs.io.read_morph_data(f)

        elif f.endswith('annot'):
            raw = fs.io.read_annot(f)[0]

        elif f.endswith('label'):
            f = fs.io.read_label(f)
        
        # Numpy case
        elif f.endswith('.npy') or f.endswith('.npz'):
            raw = np.load(f)
        
        # Otherwise, try to load with nibabel
        else:

            data = nib.load(f)
            
            # If Gifti, handle special
            if isinstance(data, GiftiImage):
                raw = np.asarray([arr.data for arr in data.darrays]).T.squeeze()
            
            # Special case if load with nibabel and index_slice is
            # passed, load from the dataobj
            elif index_slice is not None:
                
                # Return directly here
                return data.dataobj[index_slice]
            
            # Otherwise just load full
            else:
                raw = data.get_fdata()

    # If a Parc object
    elif hasattr(f, 'get_parc'):
        raw = f.get_parc(copy=True)

    else:
        raise RuntimeError(f'Unable to load passed {f}')

    # Return either array or index'ed array
    if index_slice is not None:
        return raw[index_slice]

    return raw

def _apply_template(subject, template_path, contrast=None):
    
    # Copy template path under f
    f = template_path 

    # Replace subject
    f = f.replace('SUBJECT', str(subject))
    
    # Optionally replace contrast
    if contrast is not None:
        f = f.replace('CONTRAST', str(contrast))
    
    return f

def _load_subject(subject, contrast, template_path, mask=None,
                  index_slice=None, _print=print):

    # Get the specific path for this subject based on the passed template
    f = _apply_template(subject=subject,
                        template_path=template_path,
                        contrast=contrast)

    _print('Loading:', f, level=2)
    raw = load(f, index_slice=index_slice)

    # If no mask, just flatten / make sure 1D
    if mask is None:
        flat_data = raw.flatten()
    
    # If mask, then use to create 1D array
    else:
        flat_data = raw[mask == 1]

    return flat_data
 
def get_data(subjects, template_path, contrast=None, mask=None,
             index_slice=None, zero_as_nan=False,
             n_jobs=1, verbose=1, _print=None):
    '''This method is designed to load data saved in a particular way,
    specifically where each subject / participants data is saved seperately.

    This method of loading uses a template path.

    Parameters
    ----------
    subjects : array-like
        A list or array-like with the names of the subjects to load, where
        the names correspond in some way to the way the way the subject's data
        is saved. This correspondence is specified in template_path.

    template_path : str
        A str indicating the template form for how a single
        subjects data should be loaded, where SUBJECT will be
        replaced with that subjects name, and optionally CONTRAST will
        be replaced with the contrast name. 

        For example, to load subject X's contrast Y saved under
        'some_loc/X_Y.nii.gz'
        the template_path would be: 'some_loc/SUBJECT_CONTRAST.nii.gz'.

        Note that the use of a CONTRAST argument is optional.

    contrast : str, optional
        The name of the contrast, used along with the template
        path to define where to load data.

        If passed None, then it is assumed that CONTRAST
        is not present in the template_path and will be ignored.

        ::

            default = None

    mask : str, numpy array or None
        After data is loaded, it can optionally be
        masked. By default, this parameter is set to None.
        If None, then the subjects data will be flattened.
        If passed a str, it will be assumed to be the location of a mask
        in which to load, where the shape
        of the mask should match the data.

        When passing a mask, values of True or 1 indicate that the
        value should be kept when loading data.

        If passed None, this option is ignored and the
        full passed data is used.

        ::

            default = None

    index_slice : slices, tuple of slices, or None
        You may optional pass index slicing here.
        The typical benefit of index_slicing over masking
        is where data to load will be read with nibabel
        (e.g., saved as .nii or .mgz) and can therefore use an index_slice
        to load in only the data of interest. 

        For example let's say the shape of data is 5 x 20 on
        the disk, but you only need the data in the first 20, e.g.,
        so with traditional slicing that is [0], in this case by using the
        mask you have to load the full data from the disk, then slice it.
        But if you pass `slice(0)` here, then this accomplishes the same thing,
        but only loads the requested slice from the disk.

        You must use python keyword `slice` for complex slicing. For example
        in the example above you may pass either `(0)` or `slice(0)`,
        but if you wanted something more complex, e.g. passing
        something like my_array[1:5:2, ::3] you should pass
        (slice(1,5,2), slice(None,None,3)) here.

        If passed None, this option is ignored and the
        full passed data is used.

        ::

            default = None

    zero_as_nan : bool, optional
        Often in neuroimaging data, NaN values
        are encoded as 0's (no i'm not sure why either).
        If this flag is set to True, then any 0's
        found in the loaded data will be replaced with NaN.
        
        ::

            default = False

    n_jobs : int
        The number of jobs to try and use for loading data.

    verbose : int
        By default this value is 1. This parameter
        controls the verbosity of this function.

        If -1, then no message at all will be printed.
        If 0, only warnings will be printed.
        If 1, general status updates will be printed.
        If >= 2, full verbosity will be enabled.
    '''
    
    # Keep track of loading time
    start_time = time.time()
    
    # Either init _print or use passed value
    _print = _get_print(verbose, _print=_print)
    
    # Can pass mask as None, file loc, or numpy array
    if mask is not None:
        if isinstance(mask, str):
            mask = nib.load(mask).get_fdata()

    # Single core case
    if n_jobs == 1:

        subjs_data = []
        for subject in subjects:
            subjs_data.append(_load_subject(subject, contrast, template_path,
                                            mask=mask, index_slice=index_slice,
                                            _print=_print))

    # Multi-proc case
    else:
        subjs_data = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_load_subject)(
                subject=subject,
                contrast=contrast,
                template_path=template_path,
                mask=mask,
                index_slice=index_slice,
                _print=_print) for subject in subjects)

    # Return array with shape as the number of subjects by the sum of the mask
    data = np.stack(subjs_data)

    _print('Loaded:', data.shape, 'in',
            time.time() - start_time,
            'seconds.', level=1)

    # Process zero_as_nan flag
    if zero_as_nan:
        _print(f'Setting {len(data[data == 0])} == 0 data points to NaN.')
        data[data == 0] = np.nan

    return data

def reverse_mask_data(data, mask):
    '''This function is used to reverse applying a mask.
    If mask is None, return passed data as is.'''

    if mask is None:
        return data

    # If location to nib file
    if isinstance(mask, str):
        raw_mask = nib.load(mask)
        mask = raw_mask.get_fdata()
        mask_affine = raw_mask.affine
    
    # If already mask array
    else:
        mask_affine = None

    # If multiple subjects
    if len(data.shape) == 2:
        
        projected = []
        for subj in data:
            projected.append(_project_single_subj(subj, mask, mask_affine))

        return projected

    # If single subject
    return _project_single_subj(data, mask, mask_affine)

def _project_single_subj(subj, mask, mask_affine):

    # Make copy of mask to fill
    proj_subj = mask.copy()
    proj_subj[mask == 1] = subj
    
    # Wrap in nifti image if needed
    if mask_affine is not None:
        proj_subj = nib.Nifti1Image(proj_subj, affine=mask_affine)

    return proj_subj

def get_overlap_subjects(df, template_path, contrast=None, verbose=1, _print=None):
    '''Helper function to be used when working with template_path style saved
    data in order to compute an overlapping set of subjects between a dataframe
    and a template path.

    Parameters
    ----------
    subjects : array-like
        A list or array-like with the names of the subjects to load, where
        the names correspond in some way to the way the way the subject's data
        is saved. This correspondence is specified in template_path.

    template_path : str
        A str indicating the template form for how a single
        subjects data should be loaded (or in this case located),
        where SUBJECT will be replaced with that subjects name,
        and optionally CONTRAST will be replaced with the contrast name. 

        For example, to load subject X's contrast Y saved under
        'some_loc/X_Y.nii.gz'
        the template_path would be: 'some_loc/SUBJECT_CONTRAST.nii.gz'.

        Note that the use of a CONTRAST argument is optional.

    contrast : str, optional
        The name of the contrast, used along with the template
        path to define where to load data.

        If passed None, then it is assumed that CONTRAST
        is not present in the template_path and will be ignored.

        ::

            default = None

    verbose : int
        By default this value is 1. This parameter
        controls the verbosity of this function.

        If -1, then no message at all will be printed.
        If 0, only warnings will be printed.
        If 1, general status updates will be printed.
        If >= 2, full verbosity will be enabled.

    '''

    # Either init _print or use passed value
    _print = _get_print(verbose, _print=_print)

    _print('Passed df with shape', df.shape, level=1)
    _print('Determining valid overlap subjects', level=1)
    
    # Only include subject if found as file
    all_subjects = [s for s in df.index if 
                    os.path.exists(_apply_template(subject=s,
                                                   template_path=template_path,
                                                   contrast=contrast))]
    _print('Found', len(all_subjects), 'subjects with data', level=1)
    
    # Print missing subjects if high enough verbose
    missing_subjects = [s for s in df.index if s not in all_subjects]
    _print('Missing:', missing_subjects, level=2)

    return all_subjects


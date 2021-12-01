import nibabel as nib
import numpy as np
import os
import time
from joblib import Parallel, delayed
from nibabel import freesurfer as fs
from nibabel import GiftiImage

from ..misc.print import _get_print

def _gifti_to_np(data):
    return np.asarray([arr.data for arr in data.darrays]).T.squeeze()

def load(f, index_slice=None, dtype=None):
    '''Smart loading function for neuro-imaging type data.

    Parameters
    ----------
    f : str, :func:`arrays<numpy.array>` or other
        The location / file path of the data to load passed
        as a str, or can accept directly numpy :func:`arrays<numpy.array>`
        or objects from nibabel, ect...

    index_slice : slices, tuple of slices, or :data:`None`, optional
        You may optional pass index slicing here.
        The typical benefit of index_slicing over masking in a later
        step is where data to load will be read with nibabel
        (e.g., saved as .nii or .mgz) and can therefore use an index_slice
        to load in only the data of interest. 

        For example let's say the shape of data is 5 x 20 on
        the disk, but you only need the data in the first 20, e.g.,
        so with traditional slicing that is [0], in this case by using the
        mask you have to load the full data from the disk, then slice it.
        But if you pass :python:`slice(0)` here, then this accomplishes the same thing,
        but only loads the requested slice from the disk.
        
        You must use python keyword `slice` for complex slicing. For example
        in the example above you may pass either :python:`(0)` or :python:`slice(0)`,
        but if you wanted something more complex, e.g. passing
        something like :python:`my_array[1:5:2, ::3]` you should pass
        :python:`(slice(1,5,2), slice(None,None,3))` here.

        ::

            default = None

    dtype : str or None, optional
        The data type in which to cast loaded data to.
        If left as None, keep original data type, otherwise
        can pass as a string name of datatype to cast to.
        For example, 'float32' to cast to 32 bit floating point
        precision.

        ::

            default = None

    Returns
    ---------
    data : :class:`array<numpy.ndarray>` or :data:`None`
        Returned is the requested data as a :func:`numpy array<numpy.array>`,
        or if passed :data:`None`, then :data:`None` will be returned. The shape
        of the returned data will be the shape of the data array portion of
        whatever data is passed with extra singular dimensions removed
        (i.e., np.squeeze is applied). If passed an index slice, only
        the portion of the data specified by the index slice will be loaded
        and returned.

        Note: The data type of the returned data will depend on how the
        data to be loaded is saved.

    Examples
    ---------
    Data can be loaded by simply passing a file location
    to load. For example, we can load just an :func:`array<numpy.array>` of ones:

    .. ipython:: python

        from neurotools.loading import load
        data = load('data/ones.nii.gz')
        data.shape

    This function is likewise robust to being passed non-file paths, for
    example passing an already defined :func:`array<numpy.array>` to load:

    .. ipython:: python

        import numpy as np

        data = np.ones((5, 3, 1))
        load(data).shape

    Note here that load will also apply a squeeze function to any loaded data,
    getting rid of any used dimensions.

    Next, let's look at an example where we pass a specific index slice.

    .. ipython:: python

        data = np.random.random((5, 3))
        
        # Let's say we want to select columns 0 and 1
        data[:, 0:2]
        
        # It's a bit more awkward but we pass:
        load(data, index_slice=(slice(None), slice(0, 2)))

    '''

    def _proc(data):
        '''Helper func, squeezes data and returns
        with correct data type'''

        if dtype is not None:
            return np.squeeze(data.astype(dtype))
        
        return np.squeeze(data)
    
    # If already numpy array, return of as is
    if type(f) is np.ndarray:
        raw = f.copy()
    
    # If gifti already
    elif isinstance(f, GiftiImage):
        raw = _gifti_to_np(f)
    
    # If nifti like
    elif isinstance(f, (nib.Cifti2Image, nib.Nifti1Image)):
        raw = f.get_fdata()

    # Keep as None if None
    elif f is None:
        return None

    # TODO add case for loading .mat files
    # from scipy.io import loadmat
    # ex: lh_d = loadmat(lh_loc)['parcels'].squeeze()
    
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
            raw = fs.io.read_label(f)
        
        # Numpy case
        elif f.endswith('.npy') or f.endswith('.npz'):
            raw = np.load(f)
        
        # Otherwise, try to load with nibabel
        else:

            data = nib.load(f)
            
            # If Gifti, handle special
            if isinstance(data, GiftiImage):
                raw = _gifti_to_np(data)
            
            # Special case if load with nibabel and index_slice is
            # passed, load from the dataobj
            elif index_slice is not None:
                
                # Return directly here
                return _proc(data.dataobj[index_slice])
            
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
        return _proc(raw[index_slice])

    return _proc(raw)

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
                  index_slice=None, dtype=None, _print=print):

    # Get the specific path for this subject based on the passed template
    f = _apply_template(subject=subject,
                        template_path=template_path,
                        contrast=contrast)

    _print('Loading:', f, level=2)
    raw = load(f, index_slice=index_slice, dtype=dtype)

    # If no mask, just flatten / make sure 1D
    if mask is None:
        flat_data = raw.flatten()
    
    # If mask, then use to create 1D array
    else:
        flat_data = raw[mask == 1]

    return flat_data
 
def _load_mask(mask):

    # If mask is None, return None, None
    if mask is None:
        return None, None

    # Check to see if mask already has affine
    # e.g., is already a NiftiImage
    if hasattr(mask, 'affine'):
        return _load_check_mask(mask, getattr(mask, 'affine'))
    
    # If the mask is a str - try to load it
    # with nibabel first, to check for affine
    if isinstance(mask, str):
        try:
            raw_mask = nib.load(mask)
            
            # If loaded with nibabel and has affine, return
            # data and affine
            if hasattr(raw_mask, 'affine'):
                return _load_check_mask(raw_mask, getattr(raw_mask, 'affine'))
        
        # All other cases, skip
        except:
            pass

    # If reaches here, then no affine
    # in this case, just load, then return as data with None affine
    return _load_check_mask(mask, None)

def _load_check_mask(mask, affine):

    mask_data = load(mask)
    n_unique = len(np.unique(mask_data))

    if n_unique != 2:
        raise RuntimeError(f'Passed mask must have only 2 unique values! Passed mask with {n_unique} unique values.')

    return mask_data.astype('bool'), affine

def load_data(subjects, template_path, contrast=None, mask=None,
              index_slice=None, zero_as_nan=False, nan_as_zero=False,
              dtype=None, n_jobs=1, verbose=1, _print=None):
    '''This method is designed to load data saved in a particular way,
    specifically where each subject / participants data is saved seperately.

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

        Note: The use of CONTRAST within the template path is optional, but
        this parameter is not.

    contrast : str or :data:`None`, optional
        The name of the contrast, used along with the template
        path to define where to load data.
        If passed :data:`None`, then it is assumed that CONTRAST
        is not present in the template_path and will be ignored.

        ::

            default = None

    mask : str, :func:`array<numpy.array>` or :data:`None`, optional
        After data is loaded, it can optionally be
        masked according to a specific :func:`array<numpy.array>`.

        If :data:`None`, and the data to be loaded is multi-dimensional,
        then the data will be flattened by default, e.g., if each data 
        point originally has shape 10 x 2, the flattened shape will be 20.
       
        If passed a str, it will be assumed to be the location of a mask
        in which to load, where the shape of the mask should match the data.

        When passing a mask, either by location or by :func:`array<numpy.array>` directly,
        values set to either 1 or True indicate the value should be kept,
        whereas values of 0 or False indicate the values at that location should
        be discarded.

        If a mask is used, see function :func:`funcs.reverse_mask_data`
        for reversing masked data (e.g., an eventual statistical output)
        back into it's pre-masked state.

        ::

            default = None

    index_slice : slices, tuple of slices, or :data:`None`, optional
        You may optional pass index slicing here.
        The typical benefit of index_slicing over masking
        is where data to load will be read with nibabel
        (e.g., saved as .nii or .mgz) and can therefore use an index_slice
        to load in only the data of interest. 

        For example let's say the shape of data is 5 x 20 on
        the disk, but you only need the data in the first 20, e.g.,
        so with traditional slicing that is [0], in this case by using the
        mask you have to load the full data from the disk, then slice it.
        But if you pass :python:`slice(0)` here, then this
        accomplishes the same thing,
        but only loads the requested slice from the disk.

        You must use python keyword `slice` for complex slicing. For example
        in the example above you may pass either :python:`(0)` or :python:`slice(0)`,
        but if you wanted something more complex, e.g. passing
        something like :python:`my_array[1:5:2, ::3]` you should pass
        :python:`(slice(1,5,2), slice(None,None,3))` here.

        If passed :data:`None` (the default), this option is
        ignored and the full data avaliable is loaded.

        ::

            default = None

    zero_as_nan : bool, optional
        Often in neuroimaging data, NaN values
        are encoded as 0's (I'm not sure why either).
        If this flag is set to True, then any 0's
        found in the loaded data will be replaced with NaN.
        
        ::

            default = False

    nan_as_zero : bool, optional
        As an alternative to zero_as_nan, can
        instead set any NaNs found to zeros.
        Note, if this is True then zero_as_nan can
        not also be True.

        ::

            default = False

    dtype : str or None, optional
        The data type in which to cast loaded data to.
        If left as None, keep original data type, otherwise
        can pass as a string name of datatype to cast to.
        For example, 'float32' to cast to 32 bit floating point
        precision.

        ::

            default = None

    n_jobs : int, optional
        The number of threads to use when loading data.

        Note: Often n_jobs will refer to number of separate processors,
        but in this case it refers to threads,
        which are not limited by the number of cores avaliable.

        ::

            default = 1

    verbose : int, optional
        This parameter controls the verbosity of this function.

        - If -1, then no message at all will be printed.
        - If 0, only warnings will be printed.
        - If 1, general status updates will be printed.
        - If >= 2, full verbosity will be enabled.

        ::

            default = 1

    Returns
    --------
    data : :class:`array<numpy.ndarray>`
        Loaded data across all specified subjects is returned
        as a 2D numpy :func:`array<numpy.array>` (shape= subj x data) where the first
        dimension is subject and the second is a single dimension representation
        of that subjects data.

    Examples
    ----------
    Consider the following simplified example where we are interested in loading and
    concatenating the data from three subjects: 'subj1', 'subj2' and subj3'.

    .. ipython:: python
        
        import os
        from neurotools.loading import load, load_data

        # Data are saved in separate folders
        subjs = os.listdir('data/ex1')
        subjs

        # Within each subject's directory is single data file
        # E.g., loading the first subjects data
        load('data/ex1/subj1/data.nii.gz')

        # Now we can load all subjects with the load_data method
        all_data = load_data(subjects=subjs,
                             template_path='data/ex1/SUBJECT/data.nii.gz')

        all_data.shape

    Additional arguments can of course be added allowing for more flexibility.
    Consider an additional case below:

    .. ipython:: python

        # We can add a mask, keeping only
        # the first and last data point
        mask = np.zeros(10)
        mask[0], mask[9] = 1, 1

        all_data = load_data(subjects=subjs,
                             template_path='data/ex1/SUBJECT/data.nii.gz',
                             mask=mask)

    '''

    # Error if both set
    if zero_as_nan and nan_as_zero:
        raise RuntimeError('zero_as_nan and nan_as_zero cannot both be True.')

    # Keep track of loading time
    start_time = time.time()
    
    # Either init _print or use passed value
    _print = _get_print(verbose, _print=_print)
    
    # Can pass mask as None, file loc, or numpy array
    mask, _ = _load_mask(mask)

    # Single core case
    if n_jobs == 1:

        subjs_data = []
        for subject in subjects:
            subjs_data.append(_load_subject(subject, contrast, template_path,
                                            mask=mask, index_slice=index_slice,
                                            dtype=dtype,
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
                dtype=dtype,
                _print=_print) for subject in subjects)

    # Return array with shape as the number of subjects by the sum of the mask
    data = np.stack(subjs_data)

    _print('Loaded data with shape:', data.shape, 'in',
           time.time() - start_time,
           'seconds.', level=1)

    # Process zero_as_nan flag
    if zero_as_nan:
        _print(f'Setting {len(data[data == 0])} == 0 data points to NaN.')
        data[data == 0] = np.nan

    # Process nan_as_zero flag
    if nan_as_zero:
        _print(f'Setting {len(data[np.isnan(data)])} == NaN data points to 0.')
        data[np.isnan(data)] = 0

    return data

def reverse_mask_data(data, mask):
    '''This function is used to reverse the application of a mask
    to a data array.

    Parameters
    ------------
    data : :class:`numpy.ndarray` or list of
        Some data in which to transform from either
        a single dimensional :func:`array<numpy.array>` of data,
        or a 2D :func:`array<numpy.array>`
        of data with shape = subjs x data, back to the original shape
        of the data pre-masking. Instead of a 2D :func:`array<numpy.array>`, a list
        of single :func:`arrays<numpy.array>` can be passed as well.

    mask : loc or :class:`numpy.ndarray`
        The location of, or a :func:`array<numpy.array>` mask, where
        (1/True=keep, 0/False=discard) in which to reverse the
        application of. If the passed mask is either
        a :class:`Nifti1Image<nibabel.nifti1.Nifti1Image>` with an affine, or saved as a nifti,
        then as a part of the reverse mask process, the data
        will be converted into a :class:`Nifti1Image<nibabel.nifti1.Nifti1Image>` image.

    Returns
    --------
    reversed_data : :class:`numpy.ndarray`, :class:`Nifti1Image<nibabel.nifti1.Nifti1Image>` or list of
        In the case that the mask is passed in a format that
        provides an affine, data will be returned in :class:`Nifti1Image<nibabel.nifti1.Nifti1Image>`
        format. If instead the mask is passed as a :func:`array<numpy.array>` or
        location of a saved data array without affine information,
        then the data will be returned as a :func:`numpy array<numpy.array>`.

        If either a list of data :func:`arrays<numpy.array>` is passed, or data is passed
        as a 2D :func:`array<numpy.array>` with shape = subjs x data, then the returned
        reversed data will be a list with each element corresponding
        to each passed subject's reversed data.

    Examples
    ----------

    Consider the following brief example:

    .. ipython:: python

        import numpy as np
        from neurotools.loading import reverse_mask_data

        data = np.random.random((4, 2))
        mask = np.array([1, 0, 0, 1])
        
        reversed_data = reverse_mask_data(data, mask)
        reversed_data

        reversed_data[0].shape

    Or in the case where the mask is a :class:`Nifti1Image<nibabel.nifti1.Nifti1Image>`:

    .. ipython:: python
        
        import nibabel as nib

        # Convert mask to nifti
        mask = nib.Nifti1Image(mask, affine=np.eye(4))

        reversed_data = reverse_mask_data(data, mask)
        reversed_data
        
        # Closer look
        reversed_data[0].get_fdata()
        reversed_data[0].affine

    '''

    if mask is None:
        return None

    # Can pass mask as None, file loc, or numpy array
    mask, mask_affine = _load_mask(mask)

    # If multiple subjects
    if len(data.shape) == 2 or isinstance(data, list):
        
        projected = []
        for subj in data:
            projected.append(_project_single_subj(subj, mask, mask_affine))

        return projected

    # If single subject
    return _project_single_subj(data, mask, mask_affine)

def _project_single_subj(subj, mask, mask_affine):

    # Make copy of mask to fill with same dtype as subj's data
    proj_subj = mask.copy().astype(subj.dtype)
    proj_subj[mask == 1] = subj
    
    # Wrap in nifti image if needed
    if mask_affine is not None:
        proj_subj = nib.Nifti1Image(proj_subj, affine=mask_affine)

    return proj_subj

def get_overlap_subjects(df, template_path=None, contrast=None,
                         data_df=None, verbose=1, _print=None):
    '''Helper function to be used when working with template_path style saved
    data in order to compute an overlapping set of subjects between a dataframe
    and a template path, or alternatively between two dataframes.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        A :class:`pandas.DataFrame` that is indexed by the names of the
        subjects in which to overlap.
        Where the names correspond in some way to the
        way the way the subject's data is saved.

    template_path : str, optional
        A str indicating the template form for how a single
        subjects data should be loaded (or in this case located),
        where SUBJECT will be replaced with that subjects name,
        and optionally CONTRAST will be replaced with the contrast name. 

        For example, to load subject X's contrast Y saved under
        'some_loc/X_Y.nii.gz'
        the template_path would be: 'some_loc/SUBJECT_CONTRAST.nii.gz'.

        Note that the use of a CONTRAST argument is optional.

        Note that if this parameter is passed, then data_df will
        be ignored!

        ::

            default = None

    contrast : str, optional
        The name of the contrast, used along with the template
        path to define where to load data.

        If passed :data:`None` then it is assumed that CONTRAST
        is not present in the template_path and will be ignored.

        Note that if this parameter is passed, then data_df will
        be ignored! This parameter is used only with the template_path option.

        ::

            default = None

    data_df : :class:`pandas.DataFrame` or :data:`None`, optional
        Optionally specify a :class:`pandas.DataFrame`, indexed by subject, in which
        to overlap with `df`, INSTEAD of a template_path and / or contrast set of arguments.
        Explicitly, if specifying a data_df, a template_path should not be passed!

        ::

            default = None

    verbose : int, optional
        By default this value is 1. This parameter
        controls the verbosity of this function.

        If -1, then no message at all will be printed.
        If 0, only warnings will be printed.
        If 1, general status updates will be printed.
        If >= 2, full verbosity will be enabled.

        ::

            default = 1

    '''

    # Either init _print or use passed value
    _print = _get_print(verbose, _print=_print)

    _print('Passed df with shape', df.shape, level=1)
    _print('Determining valid overlap subjects', level=1)
    
    # Only include subject if found as file
    if template_path is not None:
        all_subjects = [s for s in df.index if 
                        os.path.exists(_apply_template(subject=s,
                                                       template_path=template_path,
                                                       contrast=contrast))]

    # Unless computing overlap with data df
    else:
        all_subjects = [s for s in df.index if s in data_df.index]

    _print('Found', len(all_subjects), 'subjects with data.', level=1)
    
    # Print missing subjects if high enough verbose
    missing_subjects = [s for s in df.index if s not in all_subjects]
    _print('Missing:', missing_subjects, level=2)

    return all_subjects


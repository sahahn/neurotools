
import time
import sys
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from joblib import Parallel, delayed

from ..random.permute_blocks import block_permutation, get_auto_vg

def _self_inverse(arr):
    return arr @ np.linalg.pinv(arr)

def _get_contrast(tested_vars, confounding_vars):
    
    dtype = dtype=tested_vars.dtype.name
    
    tv_s = tested_vars.shape[1]
    cv_s = confounding_vars.shape[1]

    contrast = np.vstack([np.eye(tv_s, dtype=dtype),
                          np.zeros((cv_s, tv_s), dtype=dtype)])
    
    return contrast

def _get_perm_matrix(permutation_structure, random_state, intercept_test=False):

    # If intercept test, ignore permutation structure,
    # and the permutation is just eye with random 1's and -1's.
    if intercept_test:
        flip_vals = (check_random_state(random_state).randint(2, size=(len(permutation_structure))) * 2) - 1
        return np.eye(len(permutation_structure)) * flip_vals
    
    # If passed in variance group 1D form
    if len(permutation_structure.shape) == 1:
        neg_ones = -np.ones(len(permutation_structure))
        p_struc = np.stack([neg_ones, permutation_structure], axis=1)
    
    # If passed as blocks
    else:
        p_struc = permutation_structure
    
    # Gen permutation
    p_set = block_permutation(np.eye(len(p_struc), dtype='bool'),
                              blocks=p_struc,
                              random_state=random_state)

    return p_set

def _run_permutation_chunks(run_perm_func, original_scores,
                            thread_id, target_vars, rz, hz,
                            input_matrix, variance_groups, drm,
                            contrast, n_perm_chunk, n_perm, random_state,
                            permutation_structure=None, verbose=0,
                            use_z=False, two_sided_test=True,
                            intercept_test=False,
                            use_special_tf=False):

    # If the special tensorflow case, and the first job
    if use_special_tf and thread_id == 1:
        from ._tf_run_permutation import run_permutation, cast_input_to_tensor

        # Override the run_perm_func with the tensorflow version
        run_perm_func = run_permutation

        # Cast input to tensor
        target_vars, rz, hz, input_matrix, drm, contrast =\
            cast_input_to_tensor(target_vars, rz, hz, input_matrix,
                                    drm, contrast, dtype=target_vars.dtype)

    # Init starting vars
    t0 = time.time()
    h0_vmax_part = np.empty(n_perm_chunk)
    scores_as_ranks_part = np.zeros(target_vars.shape[1])

    # Run each permutation
    for i in range(n_perm_chunk):
        
        # Generate permutation on the fly
        p_set = _get_perm_matrix(permutation_structure, random_state+i, intercept_test)

        # Get v stats for this permutation
        perm_scores = run_perm_func(
            p_set=p_set, target_vars=target_vars, rz=rz, hz=hz,
            input_matrix=input_matrix, variance_groups=variance_groups,
            drm=drm, contrast=contrast, use_z=use_z)
        
        # Convert perm scores to to absolute values if two sides
        if two_sided_test:
            perm_scores = np.fabs(perm_scores)

        # Add max v stat
        h0_vmax_part[i] = np.nanmax(perm_scores)

        # Find the rank of the original scores in h0_part,
        # basically just compares the vmax and adds 1 if the
        # original score is higher, so e.g., if original score
        # higher than all permutations, will eventually have the highest rank.
        scores_as_ranks_part += (h0_vmax_part[i] < original_scores)
    
        # Per-step verbosity - from nilearn func
        if verbose > 0:
            step = 11 - min(verbose, 10)
            if i % step == 0:
                if n_perm == n_perm_chunk:
                    crlf = "\r"
                else:
                    crlf = "\n"
                percent = float(i) / n_perm_chunk
                percent = round(percent * 100, 2)
                dt = time.time() - t0
                remaining = (100. - percent) / max(0.01, percent) * dt
                sys.stderr.write(
                    "Job #%d, processed %d/%d permutations "
                    "(%0.2f%%, %i seconds remaining)%s"
                    % (thread_id, i, n_perm_chunk, percent, remaining, crlf))

    if verbose > 0:
        print(f'Job finished in {time.time() - t0}')

    return scores_as_ranks_part, h0_vmax_part

def _proc_input(tested_vars, confounding_vars):
    
    # Stack input vars
    input_vars = np.hstack([tested_vars, confounding_vars])
    
    # Get contrasts
    contrast = _get_contrast(tested_vars, confounding_vars)
    contrast_pinv = np.linalg.pinv(contrast.T)
    I_temp = np.eye(len(contrast), dtype=tested_vars.dtype.name)
    contrast0 = I_temp - contrast @ np.linalg.pinv(contrast)
    
    # Selects X - not sure
    # what different is between just
    # using tested_vars directly
    tmpX = input_vars @ contrast_pinv
    
    # Gets svd for confounds
    tmpZ = np.linalg.svd(input_vars @ contrast0)[0]

    # Get Z and X
    Z = tmpZ[:,0:confounding_vars.shape[-1]]
    X = tmpX - Z @ np.linalg.pinv(Z) @ tmpX

    return X, Z, contrast

def _proc_n_perm_chunks(n_perm, n_jobs, use_special_tf=False, special_tf_job_split=None):

    # Special case
    if use_special_tf:

        # Set by default 75% of permutations to run on gpu
        gpu_perms = np.int(np.round(n_perm * special_tf_job_split))

        # Split rest across remaining jobs
        rest_perm_chunks = _proc_n_perm_chunks(n_perm - gpu_perms, n_jobs-1)
        
        # Combine and return
        return np.concatenate([[gpu_perms], rest_perm_chunks]).astype('int')

    # Otherwise, generate n_perm_chunks and return those
    if n_perm > n_jobs:
        n_perm_chunks = np.asarray([n_perm / n_jobs] * n_jobs, dtype=int)

        # Distribute rest to remaining
        remaining = n_perm % n_jobs
        for r in range(remaining):
            n_perm_chunks[r] += 1
    else:
        n_perm_chunks = np.ones(n_perm, dtype=int)

    return n_perm_chunks.astype('int')

def _proc_dtype(vars):

    # Passed as int cases
    if 'int' in vars.dtype.name:

        if vars.dtype.name == 'int64':
            return vars.astype('float64')

        # Default other case
        return vars.astype('float32')

    return vars

def _nan_check(vars):
    if np.isnan(vars).any():
        raise RuntimeError('Currently no missing data is supported in tested_vars, target_vars or confounds_vars.')

def _proc_min_vg_size(min_vg_size, confounding_vars, tested_vars,
                     target_vars, permutation_structure, variance_groups):

    # Calc sizes
    u_groups, u_vg_cnts = np.unique(variance_groups, return_counts=True)

    print(f'Dropping {len(u_groups[u_vg_cnts < min_vg_size])} data points for < min_vg_size = {min_vg_size}', flush=True)

    # Get group index to keep
    to_keep = u_groups[u_vg_cnts >= min_vg_size]

    # Get index of subjects to keep
    i = np.where(np.isin(variance_groups, to_keep))[0]

    return (confounding_vars[i], tested_vars[i], target_vars[i],
            permutation_structure[i], variance_groups[i])

def _process_permuted_v_base(tested_vars,
                             target_vars,
                             confounding_vars,
                             permutation_structure,
                             n_perm=100,
                             two_sided_test=True,
                             within_grp=True,
                             random_state=None,
                             use_tf=False,
                             n_jobs=1,
                             dtype=None,
                             use_z=False,
                             demean_confounds=True,
                             min_vg_size=None):

    # Make sure input okay / right types
    if n_perm < 0:
        raise RuntimeError('n_perm must be 0 or greater.')
    if not n_jobs > 0:
        raise RuntimeError('n_jobs must be 1 or more.')
    if not isinstance(two_sided_test, bool):
        raise RuntimeError('two_sided_test must be passed a bool True / False.')
    if not isinstance(within_grp, bool):
        raise RuntimeError('within_grp must be passed a bool True / False.')

    # Get rng instance from passed random_state
    rng = check_random_state(random_state)

    # Special case of n_jobs > 1 and use_tf
    special_tf_job_split = .75
    if isinstance(use_tf, float):
        special_tf_job_split = use_tf
        use_tf = True

        if special_tf_job_split >= 1 or special_tf_job_split <= 0:
            raise RuntimeError('use_tf must be between 0 and 1 or True / False.')

    # Error if both use_tf and request calculate z
    if use_tf and use_z:
        raise RuntimeError('use_z is not currently supported with tensorflow version.')

    # If passed as DataFrame or Series cast to array
    if isinstance(tested_vars, (pd.DataFrame, pd.Series)):
        tested_vars = np.array(tested_vars)
    if isinstance(confounding_vars, (pd.DataFrame, pd.Series)):
        confounding_vars = np.array(confounding_vars)
    if isinstance(permutation_structure, (pd.DataFrame, pd.Series)):
        permutation_structure = np.array(permutation_structure)

    # Make sure input shape / dims are correct
    if len(tested_vars.shape) == 1:
        tested_vars = np.expand_dims(tested_vars, axis=1)
   
    if tested_vars.shape[1] > 1:
        raise RuntimeError('You may not pass more than one vars to test!')

    if len(confounding_vars.shape) == 1:
        confounding_vars = np.expand_dims(confounding_vars, axis=1)

    # Make sure the same len for main passed vars
    if (len(tested_vars) != len(target_vars)) or (len(confounding_vars)) !=\
        len(permutation_structure) or (len(confounding_vars) != len(tested_vars)):
        raise RuntimeError('The passed lengths for tested_vars, target_vars, ' \
                           'confounding_vars and permutation structure must all be the same!')
    
    # Check for nans
    _nan_check(confounding_vars)
    _nan_check(tested_vars)
    _nan_check(target_vars)
    _nan_check(permutation_structure)

    # Proc dtype
    confounding_vars = _proc_dtype(confounding_vars)
    tested_vars = _proc_dtype(tested_vars)
    target_vars = _proc_dtype(target_vars)

    # Check if tested_vars is intercept or not
    if np.unique(tested_vars).size == 1:
        intercept_test = True
    else:
        intercept_test = False

    # TODO make sure all the same dtype

    # Calculate variance groups from passed permutation structure
    variance_groups = get_auto_vg(permutation_structure, within_grp=within_grp)

    # Optionally, remove some data points based on too small unique variance group
    if min_vg_size is not None:
        confounding_vars, tested_vars, target_vars, permutation_structure, variance_groups =\
            _proc_min_vg_size(min_vg_size, confounding_vars, tested_vars,
                             target_vars, permutation_structure, variance_groups)

    # Optionally de-mean confounds, prevents common
    # problems, especially when passing dummy coded variables
    # so default is True
    if demean_confounds:
        confounding_vars -= confounding_vars.mean(axis=0)

    # Process dtype argument if passed
    if dtype is not None:
        tested_vars = tested_vars.astype(dtype)
        confounding_vars = confounding_vars.astype(dtype)
        target_vars = target_vars.astype(dtype)

    # Process / resid ?
    tested_resid, confounding_resid, contrast =\
        _proc_input(tested_vars, confounding_vars)

    # Get all input vars
    input_matrix = np.hstack([tested_resid, confounding_resid])

    # Pre-calculate here
    I = np.eye(len(input_matrix), dtype=input_matrix.dtype.name)
    drm = np.diag(I - _self_inverse(input_matrix))

    # Get inverse of confounding vars
    hz = _self_inverse(confounding_resid)
    rz = I - hz

    # Save some memory
    del tested_resid, tested_vars, confounding_vars, confounding_resid

    # If we are going to use tensorflow
    use_special_tf = False
    if use_tf:

        # If n_jobs is 1, base behavior just run tensorflow
        if n_jobs == 1:

            from ._tf_run_permutation import run_permutation, cast_input_to_tensor

            # Cast to tensor
            target_vars, rz, hz, input_matrix, drm, contrast =\
                cast_input_to_tensor(target_vars, rz, hz, input_matrix,
                                     drm, contrast, dtype=dtype)
        
        # Otherwise, special multi-processing case
        else:

            from ._base_run_permutation import run_permutation
            
            # Special tensorflow / cpu case
            # so we use base cpu permutation for the initial baseline
            # then use the special
            use_special_tf = True
        
    else:

        from ._base_run_permutation import run_permutation

    # Get original scores, pass permutation as None
    original_scores = run_permutation(p_set=np.eye(len(input_matrix)),
                                      target_vars=target_vars,
                                      rz=rz, hz=hz, input_matrix=input_matrix,
                                      variance_groups=variance_groups,
                                      drm=drm, contrast=contrast,
                                      use_z=use_z)

    # If two sided test, get absolute value of the scores
    original_scores_sign = None
    if two_sided_test:
        original_scores_sign = np.sign(original_scores)
        original_scores = np.fabs(original_scores)

    # A little ugly but ... for now
    return (original_scores, original_scores_sign,
            run_permutation, target_vars,
            rz, hz, input_matrix, permutation_structure, variance_groups, 
            drm, contrast, rng, intercept_test, use_special_tf, special_tf_job_split)

def permuted_v(tested_vars,
               target_vars,
               confounding_vars,
               permutation_structure,
               n_perm=100,
               two_sided_test=True,
               within_grp=True,
               random_state=None,
               use_tf=False,
               n_jobs=1,
               verbose=0,
               dtype=None,
               use_z=False,
               demean_confounds=True,
               min_vg_size=None):
    '''This function is used to perform a permutation based statistical test
    on data with an underlying exchangability-block type structure.

    Parameters
    -----------
    tested_vars : numpy array or pandas DataFrame
        Pass as an array or DataFrame, with shape
        either 1D or 2D as subjects x 1, and containing the
        values of interest in which to calculate the v-statistic
        against the passed `target_vars`, and as corrected for
        any passed `confounding_vars`.

        Note: The subject / data point order and length should match exactly
        between the first dimensions of `tested_vars`, 'target_vars`,
        `confounding_vars` and `permutation_structure`.

    target_vars : numpy array
        A 2D numpy array with shape subjects x features, containing
        the typically imaging features per subject to univariately
        calculate v-stats for.

        Note: The subject / data point order and length should match exactly
        between the first dimensions of `tested_vars`, 'target_vars`,
        `confounding_vars` and `permutation_structure`.

    confounding_vars : numpy array or pandas DataFrame

        Note: The subject / data point order and length should match exactly
        between the first dimensions of `tested_vars`, 'target_vars`,
        `confounding_vars` and `permutation_structure`.

    permutation_structure : numpy array or pandas DataFrame
        This parameter represents the underlying exchangability-block
        structure of the data passed. It is also used in order to automatically determine
        the underlying variance structure of the passed data.

        See PALM's documentation for an introduction on how to format ExchangeabilityBlocks:
        https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/PALM/ExchangeabilityBlocks

        This parameter accepts the same style input as PALM, except it is passed
        here as an array or DataFrame instead of as a file. The main requirement 
        is that the shape of the structure match the number of subjects / data points
        in the first dimension.
        
        Note: The subject / data point order and length should match exactly
        between the first dimensions of `tested_vars`, 'target_vars`,
        `confounding_vars` and `permutation_structure`.

    n_perm : int, optional
        The number of permutations to test.

        ::

            default = 0

    two_sided_test : bool, optional
        
        ::

            default = True

    Returns
    --------
    pvals : numpy array
        X
    
    original_scores : numpy array
        X 
    
    h0_vmax : numpy array:
        X

    '''

    # TODO add handle missing-ness better?
    # TODO if tested vars multiple, setup to run each seperately ... same as nilearn
    # TODO change to class based for cleaner? Whole function as a class instead
    # TODO support no confounding vars (in that case add intercept?)

    # Separate function for initial processing
    (original_scores, original_scores_sign, run_permutation,
     target_vars, rz, hz, input_matrix, permutation_structure,
     variance_groups, drm, contrast, rng, intercept_test,
     use_special_tf, special_tf_job_split) =\
        _process_permuted_v_base(tested_vars=tested_vars,
                                 target_vars=target_vars,
                                 confounding_vars=confounding_vars,
                                 permutation_structure=permutation_structure,
                                 n_perm=n_perm,
                                 two_sided_test=two_sided_test,
                                 within_grp=within_grp,
                                 random_state=random_state,
                                 use_tf=use_tf,
                                 n_jobs=n_jobs,
                                 dtype=dtype,
                                 use_z=use_z,
                                 demean_confounds=demean_confounds,
                                 min_vg_size=min_vg_size)

    # Return original scores - if no permutation
    if n_perm == 0:

        # Convert back to signed scores before returning
        if two_sided_test:
            original_scores *= original_scores_sign

        return np.asarray([]), original_scores, np.asarray([])

    # Get n_perm_chunks, or also case where is n_perm
    # is pre-generated, splits it by jon
    n_perm_chunks = _proc_n_perm_chunks(n_perm, n_jobs,
                                        use_special_tf,
                                        special_tf_job_split)

    # Submit permutations with joblib
    ret = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(_run_permutation_chunks)(
        run_perm_func=run_permutation, original_scores=original_scores,
        thread_id=thread_id+1, target_vars=target_vars,
        rz=rz, hz=hz, input_matrix=input_matrix,
        variance_groups=variance_groups, drm=drm, contrast=contrast,
        n_perm_chunk=n_perm_chunk, n_perm=n_perm,
        random_state=rng.randint(1, np.iinfo(np.int32).max - 1),
        permutation_structure=permutation_structure,
        verbose=verbose, use_z=use_z, two_sided_test=two_sided_test,
        intercept_test=intercept_test,
        use_special_tf=use_special_tf)
        for thread_id, n_perm_chunk in enumerate(n_perm_chunks))

    # Collect returned results together
    scores_as_ranks_parts, h0_vmax_parts = zip(*ret)

    # Stack together the max v stats from each permutation
    h0_vmax = np.hstack(h0_vmax_parts)

    # Combine the scores as ranks
    scores_as_ranks = np.sum(scores_as_ranks_parts, axis=0)
 
    # Convert ranks into p-values
    pvals = (n_perm + 1 - scores_as_ranks) / float(1 + n_perm)

    # Convert back to signed scores before returning
    if two_sided_test:
        original_scores *= original_scores_sign

    return pvals, original_scores, h0_vmax

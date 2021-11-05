
import time
import sys
import numpy as np
from sklearn.utils import check_random_state
from joblib import Parallel, delayed

from ..random.permute_blocks import block_permutation, get_auto_vg

def self_inverse(arr):
    return arr @ np.linalg.pinv(arr)

def get_contrast(tested_vars, confounding_vars):
    
    dtype = dtype=tested_vars.dtype.name
    
    tv_s = tested_vars.shape[1]
    cv_s = confounding_vars.shape[1]

    contrast = np.vstack([np.eye(tv_s, dtype=dtype),
                          np.zeros((cv_s, tv_s), dtype=dtype)])
    
    return contrast

def _get_perm_matrix(permutation_structure, random_state):
    
    # If passed in variance group 1D form
    if len(permutation_structure.shape):
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
                            use_z=False, two_sided_test=True, use_special_tf=False):

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
        p_set = _get_perm_matrix(permutation_structure, random_state+i)

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

    return scores_as_ranks_part, h0_vmax_part

def proc_input(tested_vars, confounding_vars):
    
    # Stack input vars
    input_vars = np.hstack([tested_vars, confounding_vars])
    
    # Get contrasts
    contrast = get_contrast(tested_vars, confounding_vars)
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
        n_perm_chunks[-1] += n_perm % n_jobs
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
               demean_confounds=True):

    # TODO add handle missing-ness better?

    if n_perm < 0:
        raise RuntimeError('n_perm must be 0 or greater.')

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
    
    # Make sure input correct
    if len(tested_vars.shape) == 1:
        tested_vars = np.expand_dims(tested_vars, axis=1)
   
    if tested_vars.shape[1] > 1:
        raise RuntimeError('You may not pass more than one vars to test!')

    if len(confounding_vars.shape) == 1:
        confounding_vars = np.expand_dims(confounding_vars, axis=1)
    
    # Check for nans
    _nan_check(confounding_vars)
    _nan_check(tested_vars)
    _nan_check(target_vars)

    # Proc dtype
    confounding_vars = _proc_dtype(confounding_vars)
    tested_vars = _proc_dtype(tested_vars)
    target_vars = _proc_dtype(target_vars)

    # TODO make sure all the same dtype

    # Optionally de-mean confounds, prevents common
    # problems, especially when passing dummy coded variables
    # so default is True
    if demean_confounds:
        confounding_vars -= confounding_vars.mean(axis=0)
    
    # Calculate variance groups from passed permutation structure
    variance_groups = get_auto_vg(permutation_structure, within_grp=within_grp)

    # Process dtype argument if passed
    if dtype is not None:
        tested_vars = tested_vars.astype(dtype)
        confounding_vars = confounding_vars.astype(dtype)
        target_vars = target_vars.astype(dtype)

    # Process / resid ?
    tested_resid, confounding_resid, contrast =\
        proc_input(tested_vars, confounding_vars)

    # Get all input vars
    input_matrix = np.hstack([tested_resid, confounding_resid])

    # Pre-calculate here
    I = np.eye(len(input_matrix), dtype=input_matrix.dtype.name)
    drm = np.diag(I - self_inverse(input_matrix))

    # Get inverse of confounding vars
    hz = self_inverse(confounding_resid)
    rz = I - hz

    # Save some memory
    del tested_resid, tested_vars, confounding_vars, confounding_resid

    if not n_jobs > 0:
        raise RuntimeError('n_jobs must be 1 or more.')

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
    if two_sided_test:
        original_scores_sign = np.sign(original_scores)
        original_scores = np.fabs(original_scores)

    # Return original scores - if no permutation
    if n_perm == 0:

        # Convert back to signed scores before returning
        if two_sided_test:
            original_scores *= original_scores_sign

        return np.asarray([]), original_scores, np.asarray([])

    # Get n_perm_chunks, or also case where is n_perm
    # is pre-generated, splits it by jon
    n_perm_chunks = _proc_n_perm_chunks(n_perm, n_jobs, use_special_tf, special_tf_job_split)

    # Submit permutations with joblib
    ret = Parallel(n_jobs=n_jobs)(delayed(_run_permutation_chunks)(
        run_perm_func=run_permutation, original_scores=original_scores,
        thread_id=thread_id+1, target_vars=target_vars,
        rz=rz, hz=hz, input_matrix=input_matrix,
        variance_groups=variance_groups, drm=drm, contrast=contrast,
        n_perm_chunk=n_perm_chunk, n_perm=n_perm,
        random_state=rng.randint(1, np.iinfo(np.int32).max - 1),
        permutation_structure=permutation_structure,
        verbose=verbose, use_z=use_z, two_sided_test=two_sided_test,
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

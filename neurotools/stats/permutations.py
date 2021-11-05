
import time
import sys
import numpy as np
import scipy
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
                            contrast, n_perm_chunk, n_perms, random_state,
                            permutation_structure=None,
                            verbose=0, use_z=False):

    # If n_perm_chunk is passed as not an int
    # then this is the case where pre-generated permutations
    # have been passed
    if hasattr(n_perm_chunk, '__iter__'):
        p_sets = n_perm_chunk
        n_perm_chunk = len(p_sets)
    else:
        p_sets = None

    # Init starting vars
    t0 = time.time()
    h0_vmax_part = np.empty(n_perm_chunk)
    scores_as_ranks_part = np.zeros(target_vars.shape[1])

    # Run each permutation
    for i in range(n_perm_chunk):
        
        # Generate permutation on the fly case
        if p_sets is None:
            p_set = _get_perm_matrix(permutation_structure, random_state+i)

        # Pre-generated permutations case
        else:
            p_set = p_sets[i]
            
            # If sparse, convert to dense
            if scipy.sparse.issparse(p_set):
                p_set = p_set.toarray()

        # Get v stats for this permutation
        perm_scores = run_perm_func(
            p_set=p_set, target_vars=target_vars, rz=rz, hz=hz,
            input_matrix=input_matrix, variance_groups=variance_groups,
            drm=drm, contrast=contrast, use_z=use_z)

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
                if n_perms == n_perm_chunk:
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

def _proc_n_perm_chunks(n_perm, n_jobs):

    # If n_perm is not an int, then split the
    # passed permutations into a list and return that
    # instead of n_perm_chunks
    if hasattr(n_perm, '__iter__'):
        return np.array_split(n_perm, n_jobs), len(n_perm)

    # Otherwise, generate n_perm_chunks and return those
    if n_perm > n_jobs:
        n_perm_chunks = np.asarray([n_perm / n_jobs] * n_jobs, dtype=int)
        n_perm_chunks[-1] += n_perm % n_jobs
    else:
        n_perm_chunks = np.ones(n_perm, dtype=int)

    return n_perm_chunks.astype('int'), int(n_perm)

def permuted_v(tested_vars, target_vars,
               confounding_vars, n_perm=30,
               permutation_structure=None,
               variance_groups=None,
               within_grp=True,
               random_state=None,
               use_tf=False,
               n_jobs=1,
               verbose=0,
               dtype=None,
               use_z=False):

    # Get rng instance from passed random_state
    rng = check_random_state(random_state)

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

    # Proc variance groups if not passed
    if variance_groups is None:
        if permutation_structure is None:
            raise RuntimeError('No variance groups or permutation structure passed! If really no variance groups, you are better off using a standard permuted_ols.')
        
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

    # If we are going to use tensorflow
    if use_tf:
        from ._tf_run_permutation import run_permutation, cast_input_to_tensor

        # Cast to tensor
        target_vars, rz, hz, input_matrix, drm, contrast =\
            cast_input_to_tensor(target_vars, rz, hz, input_matrix,
                                 drm, contrast, dtype=dtype)
        
    else:
        from ._base_run_permutation import run_permutation

    # Get original scores, pass permutation as None
    original_scores = run_permutation(p_set=np.eye(len(input_matrix)),
                                      target_vars=target_vars,
                                      rz=rz, hz=hz, input_matrix=input_matrix,
                                      variance_groups=variance_groups,
                                      drm=drm, contrast=contrast,
                                      use_z=use_z)

    # Return original scores - if no permutation
    if not hasattr(n_perm, '__iter__') and n_perm == 0:
        return np.asarray([]), original_scores, np.asarray([])

    # Get n_perm_chunks, or also case where is n_perm
    # is pre-generated, splits it by jon
    n_perm_chunks, n_perms = _proc_n_perm_chunks(n_perm, n_jobs)

    # Submit permutations with joblib
    ret = Parallel(n_jobs=n_jobs)(delayed(_run_permutation_chunks)(
        run_perm_func=run_permutation, original_scores=original_scores,
        thread_id=thread_id+1, target_vars=target_vars,
        rz=rz, hz=hz, input_matrix=input_matrix,
        variance_groups=variance_groups, drm=drm, contrast=contrast,
        n_perm_chunk=n_perm_chunk, n_perms=n_perms,
        random_state=rng.randint(1, np.iinfo(np.int32).max - 1),
        permutation_structure=permutation_structure, verbose=verbose, use_z=use_z)
        for thread_id, n_perm_chunk in enumerate(n_perm_chunks))

    # Collect returned results together
    scores_as_ranks_parts, h0_vmax_parts = zip(*ret)

    # Stack together the max v stats from each permutation
    h0_vmax = np.hstack(h0_vmax_parts)

    # Combine the scores as ranks
    scores_as_ranks = np.sum(scores_as_ranks_parts, axis=0)
 
    # Convert ranks into p-values
    pvals = (n_perms + 1 - scores_as_ranks) / float(1 + n_perms)

    # Return
    return pvals, original_scores, h0_vmax

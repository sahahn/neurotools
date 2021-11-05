import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import numpy as np

def cast_input_to_tensor(target_vars, rz, hz, input_matrix,
                         drm, contrast, dtype=None):

    return (tf.convert_to_tensor(target_vars, dtype=dtype),
            tf.convert_to_tensor(rz, dtype=dtype),
            tf.convert_to_tensor(hz, dtype=dtype),
            tf.convert_to_tensor(input_matrix, dtype=dtype),
            tf.convert_to_tensor(drm, dtype=dtype),
            tf.convert_to_tensor(contrast, dtype=dtype))

def calc_den_fast(c, cte, r):

    # Reshape and calc inverse of cte
    cte_inv = tf.linalg.inv(tf.transpose(tf.reshape(cte,
                                                    (r, r, cte.shape[-1])),
                                         perm=[2, 0, 1]))

    # Apply the contrast to extract correct pieces, and return squeezed
    return tnp.squeeze(tf.transpose((tf.transpose(c) @ cte_inv @ c), perm=[1, 2, 0]))

def fastv(input_matrix, psi, res,
          variance_groups, drm, contrast):
    '''Note, this version of fastv does not calculate df2.'''

    # Init vars
    dtype = input_matrix.dtype.name
    r = input_matrix.shape[-1]
    cte  = tnp.zeros((r ** 2, res.shape[-1]), dtype=dtype)

    # Process each variance group
    for vg in np.unique(variance_groups):
        
        # Get index of the current vg
        vg_idx = variance_groups == vg

        # Sum the diag resid matrix for this variance group
        diag_resid_sum_vg = tnp.sum(drm[vg_idx], axis=0)

        # Divide by sum-squared residuals
        weights_vg = diag_resid_sum_vg / tnp.sum(tnp.square(res[vg_idx]), axis=0)

        # Get correct piece of input matrix
        input_matrix_vg = tf.transpose(input_matrix[vg_idx]) @ input_matrix[vg_idx]

        # Perform matrix dot product between weights_vg and input_matrix_vg
        cte = cte + (tnp.dot(tnp.expand_dims(tf.reshape(input_matrix_vg, [-1]), axis=1),
                     tnp.expand_dims(weights_vg, axis=0)))

    # Calc rest of v and return
    return tf.transpose(contrast) @ psi / tnp.sqrt(
        calc_den_fast(contrast, cte, r))

def freedmanlane(p_set, target_vars, rz, hz):
    return (tf.transpose(p_set) @ rz + hz) @ target_vars

def run_permutation(p_set, target_vars, rz, hz,
                    input_matrix, variance_groups,
                    drm, contrast, use_z=False):

    if use_z:
        raise RuntimeError('use_z not supported with tensorflow.')

    # Need to convert each passed permutation to Tensor
    # w/ the same type as target
    p_set = tf.convert_to_tensor(p_set, dtype=target_vars.dtype.name)

    # Apply freeman-lane
    targets_permuted = freedmanlane(p_set, target_vars, rz, hz)
   
    # Fit GLM
    psi = tf.linalg.lstsq(input_matrix, targets_permuted, fast=True)
    res = targets_permuted - input_matrix @ psi
    del targets_permuted

    # Compute v stat
    v = fastv(input_matrix=input_matrix, psi=psi, res=res,
              variance_groups=variance_groups, drm=drm,
              contrast=contrast)
    
    # Return as numpy array for max compat
    return np.array(tnp.squeeze(v))
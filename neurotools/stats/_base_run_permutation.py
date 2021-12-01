import numpy as np
from math import atan
from scipy.special import erfc, erfcinv, betainc
import numpy as np

def g_to_z(g, df2):

    g = np.squeeze(g)
    
    # Get pos neg index
    idx = g > 0
    
    # Init
    z = np.empty(g.shape)
    
    # Fill pos then negative
    z[idx] = erfcinv(2 * palm_gcdf(-g[idx], df2[idx])) * np.sqrt(2)
    z[~idx] = -erfcinv(2 * palm_gcdf(g[~idx], df2[~idx])) * np.sqrt(2)
    
    return z

def palm_gcdf(g, df2):

    i_c = df2 == 1
    i_n = df2 > 1e7
    i_g = ~(i_c | i_n)
    
    # Init
    gcdf = np.zeros(g.shape)
    
    if i_g.any():

        # Get params
        a, b = df2[i_g] / 2, .5
        
        x = np.real(1 / (1 + np.square(g[i_g]) / df2[i_g]))

        # Apply func
        gcdf[i_g] = betainc(a, b, x) / 2
    
    i_g = g > 0 & i_g
    gcdf[i_g] = 1 - gcdf[i_g]
    
    # Is 1 case
    if i_c.any():
        gcdf[i_c] = .5 + atan(g[i_c]) / np.pi
    
    # Is above 1e7 case
    if i_n.any():
        gcdf[i_n] = erfc(-g[i_n] / np.sqrt(2)) / 2
        
    return gcdf

def freedmanlane(P, Y, Rz, Hz):
    return (P.T @ Rz + Hz) @ Y

def calc_den_fast(eC, cte, r):
    cte_reshape = cte.reshape((r, r, cte.shape[-1])).transpose(2, 0, 1)
    return np.squeeze((eC.T @ np.linalg.inv(cte_reshape) @ eC).transpose(1, 2, 0))

def fastv(input_matrix, psi, res,
          variance_groups, drm, contrast):

    # Init vars
    r = input_matrix.shape[-1]
    cte  = np.zeros((r ** 2, res.shape[-1]), dtype=input_matrix.dtype.name)

    # For each variance group
    for vg in np.unique(variance_groups):

        # Index of the current vg
        vg_idx = variance_groups == vg

        # Sum the diag resid matrix for this variance group
        diag_resid_sum_vg = np.sum(drm[vg_idx], axis=0)

        # Divide by sum-squared residuals
        weights_vg = diag_resid_sum_vg / np.sum(np.square(res[vg_idx]), axis=0)

         # Get correct piece of input matrix
        input_matrix_vg = input_matrix[vg_idx].T @ input_matrix[vg_idx]

        # Perform matrix dot product between weights_vg and input_matrix_vg
        cte = cte + (np.dot(np.expand_dims(input_matrix_vg.flatten(), axis=1),
                            np.expand_dims(weights_vg, axis=0)))

    # Calc rest of v and return
    return contrast.T @ psi / np.sqrt(
        calc_den_fast(contrast, cte, r))

def fastv_with_df(input_matrix, psi, res, variance_groups, drm, contrast):

    # Init vars
    n_vg = len(np.unique(variance_groups))
    dtype = input_matrix.dtype.name
    nT = res.shape[-1]
    W = np.zeros((n_vg, nT), dtype=dtype)
    r = input_matrix.shape[-1]
    dRmb = np.zeros((n_vg, 1), dtype=dtype)
    cte  = np.zeros((r ** 2, nT), dtype=dtype)

    # For each variance group
    for b, vg in enumerate(np.unique(variance_groups)):

        # Index of the current vg
        vg_idx = variance_groups == vg

        dRmb[b] = np.sum(drm[vg_idx], axis=0)
        W[b] = dRmb[b] / np.sum(np.square(res[vg_idx]), axis=0)
        input_matrix_vg = input_matrix[vg_idx].T @ input_matrix[vg_idx]
        cte = cte + (np.dot(np.expand_dims(input_matrix_vg.flatten(), axis=1), np.expand_dims(W[b], axis=0)))

        W[b] = W[b] * np.sum(vg_idx)

    # Calc den
    den = calc_den_fast(contrast, cte, r)
    G = contrast.T @ psi / np.sqrt(den)

    sW1 = np.sum(W, axis=0)
    bsum = np.sum(np.square((1-(W / sW1))) / dRmb, axis=0)
    df2 = (1/3) / bsum

    return G, df2

def run_permutation(p_set, target_vars, rz, hz,
                    input_matrix, variance_groups,
                    drm, contrast, use_z=False, **kwargs):

    # Make sure passed p_set has same dtype as target vars
    p_set = p_set.astype(target_vars.dtype.name)
    
    # Apply freeman-lane
    Y = freedmanlane(p_set, target_vars, rz, hz)

    # Fit GLM
    psi, _, _, _ = np.linalg.lstsq(input_matrix, Y, rcond=None)
    res = Y - input_matrix @ psi
    del Y

    # Base case, just compute v
    if not use_z:
        v = fastv(input_matrix=input_matrix, psi=psi, res=res,
                variance_groups=variance_groups, drm=drm,
                contrast=contrast)
        return np.squeeze(v)

    # Otherwise, special case, compute z as well
    v, df2 = fastv_with_df(input_matrix=input_matrix, psi=psi, res=res,
                           variance_groups=variance_groups, drm=drm,
                           contrast=contrast)

    # Calculate z from v
    z = g_to_z(v, df2)

    # Return just z instead
    return z



import  numpy as np

from ..permutations import (_get_contrast, _get_perm_matrix,
                            _proc_input, permuted_v)

def test_get_contrast_1():

    t = np.ones((10, 1))
    c = np.zeros((10, 2))

    contrast = _get_contrast(t, c)

    assert contrast.shape == (3, 1)
    assert contrast[0][0] == 1
    assert contrast[1][0] == 0
    assert contrast[2][0] == 0

def test_get_contrast_2():

    t = np.ones((10, 1))
    c = np.zeros((10, 1))

    contrast = _get_contrast(t, c)

    assert contrast.shape == (2, 1)
    assert contrast[0][0] == 1
    assert contrast[1][0] == 0

def test_get_contrast_3():

    t = np.ones((10, 1))
    c = np.zeros((10, 0))

    contrast = _get_contrast(t, c)

    assert contrast.shape == (1, 1)
    assert contrast[0][0] == 1

def test_get_perm_matrix():

    p_structure = np.array([1, 1, 1, 2, 2, 2])

    p_set = _get_perm_matrix(p_structure, None)
    
    assert 'bool' in p_set.dtype.name
    assert p_set.shape == ((6, 6))

def test_proc_input_1():

    tested_vars = np.ones((10, 1))
    confounding_vars = np.ones((10, 1))

    X, Z, _ = _proc_input(tested_vars, confounding_vars)

    np.testing.assert_almost_equal(np.sum(X), 0)
    np.testing.assert_almost_equal(np.sum(Z), -3.162277660168379)

def test_permuted_v_one_group():

    pvals, original_scores, h0_vmax =\
        permuted_v(tested_vars=np.ones((4)),
                   target_vars=np.random.random((4, 5)),
                   confounding_vars=np.random.random((4, 2)),
                   permutation_structure=np.array([1, 1, 2, 2]),
                   n_perm=10,
                   two_sided_test=True,
                   within_grp=True,
                   random_state=None,
                   use_tf=False,
                   n_jobs=1,
                   verbose=0,
                   dtype=None,
                   use_z=False,
                   demean_confounds=True,
                   min_vg_size=None)

    assert len(pvals) == 5
    assert len(original_scores) ==  5
    assert len(h0_vmax) == 10


            
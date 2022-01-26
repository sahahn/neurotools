
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

def test_permuted_v_rand_vals():

    pvals, original_scores, h0_vmax =\
        permuted_v(tested_vars=np.random.random((4)),
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

def test_permuted_v_rand_vals_z():

    pvals, original_scores, h0_vmax =\
        permuted_v(tested_vars=np.random.random((4)),
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
                   use_z=True,
                   demean_confounds=True,
                   min_vg_size=None)

    assert len(pvals) == 5
    assert len(original_scores) ==  5
    assert len(h0_vmax) == 10

def test_permuted_v_rand_vals_float32():

    pvals, original_scores, h0_vmax =\
        permuted_v(tested_vars=np.random.random((4)),
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
                   dtype='float32',
                   use_z=False,
                   demean_confounds=True,
                   min_vg_size=None)

    assert len(pvals) == 5
    assert len(original_scores) ==  5
    assert 'float32' in original_scores.dtype.name
    assert len(h0_vmax) == 10

def test_permuted_v_rand_vals_float32_demean_dif_no_perm():

    tested_vars=np.random.random((4))
    target_vars=np.random.random((4, 5))
    confounding_vars=np.random.random((4, 2))
    permutation_structure=np.array([1, 1, 2, 2])


    _, original_scores1, _ =\
        permuted_v(tested_vars=tested_vars,
                   target_vars=target_vars,
                   confounding_vars=confounding_vars,
                   permutation_structure=permutation_structure,
                   n_perm=0,
                   two_sided_test=True,
                   within_grp=True,
                   random_state=None,
                   use_tf=False,
                   n_jobs=1,
                   verbose=0,
                   dtype='float32',
                   use_z=False,
                   demean_confounds=False,
                   min_vg_size=None)

    _, original_scores2, _ =\
        permuted_v(tested_vars=tested_vars,
                   target_vars=target_vars,
                   confounding_vars=confounding_vars,
                   permutation_structure=permutation_structure,
                   n_perm=0,
                   two_sided_test=True,
                   within_grp=True,
                   random_state=None,
                   use_tf=False,
                   n_jobs=1,
                   verbose=0,
                   dtype='float32',
                   use_z=False,
                   demean_confounds=True,
                   min_vg_size=None)

    assert not np.array_equal(original_scores1, original_scores2)

def test_permuted_v_rand_vals_demean_dif_no_perm():

    tested_vars=np.random.random((4))
    target_vars=np.random.random((4, 5))
    confounding_vars=np.random.random((4, 2))
    permutation_structure=np.array([1, 1, 2, 2])


    _, original_scores1, _ =\
        permuted_v(tested_vars=tested_vars,
                   target_vars=target_vars,
                   confounding_vars=confounding_vars,
                   permutation_structure=permutation_structure,
                   n_perm=0,
                   two_sided_test=True,
                   within_grp=True,
                   random_state=None,
                   use_tf=False,
                   n_jobs=1,
                   verbose=0,
                   dtype=None,
                   use_z=False,
                   demean_confounds=False,
                   min_vg_size=None)

    _, original_scores2, _ =\
        permuted_v(tested_vars=tested_vars,
                   target_vars=target_vars,
                   confounding_vars=confounding_vars,
                   permutation_structure=permutation_structure,
                   n_perm=0,
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

    assert not np.array_equal(original_scores1, original_scores2)

def test_permuted_v_rand_vals_dtype_compare():

    tested_vars=np.random.random((4))
    target_vars=np.random.random((4, 5))
    confounding_vars=np.random.random((4, 2))
    permutation_structure=np.array([1, 1, 2, 2])

    _, original_scores1, _ =\
        permuted_v(tested_vars=tested_vars,
                target_vars=target_vars,
                confounding_vars=confounding_vars,
                permutation_structure=permutation_structure,
                n_perm=0,
                two_sided_test=True,
                within_grp=True,
                random_state=None,
                use_tf=False,
                n_jobs=1,
                verbose=0,
                dtype='float64',
                use_z=False,
                demean_confounds=True,
                min_vg_size=None)

    _, original_scores2, _ =\
        permuted_v(tested_vars=tested_vars,
                target_vars=target_vars,
                confounding_vars=confounding_vars,
                permutation_structure=permutation_structure,
                n_perm=0,
                two_sided_test=True,
                within_grp=True,
                random_state=None,
                use_tf=False,
                n_jobs=1,
                verbose=0,
                dtype='float32',
                use_z=False,
                demean_confounds=True,
                min_vg_size=None)

    # All close, but not equal
    assert np.allclose(original_scores1, original_scores2,
                        rtol=1e-02, atol=1e-05)
    assert not np.array_equal(original_scores1, original_scores2)


def test_permuted_v_rand_vals_min_vg_size():

    tested_vars=np.random.random((6))
    target_vars=np.random.random((6, 5))
    confounding_vars=np.random.random((6, 2))
    permutation_structure=np.array([1, 1, 1, 3, 3, 2])

    _, original_scores1, _ =\
        permuted_v(tested_vars=tested_vars,
                   target_vars=target_vars,
                   confounding_vars=confounding_vars,
                   permutation_structure=permutation_structure,
                   n_perm=0,
                   two_sided_test=True,
                   within_grp=True,
                   random_state=None,
                   use_tf=False,
                   use_z=False,
                   demean_confounds=True,
                   min_vg_size=None)

    _, original_scores2, _ =\
        permuted_v(tested_vars=tested_vars,
                   target_vars=target_vars,
                   confounding_vars=confounding_vars,
                   permutation_structure=permutation_structure,
                   n_perm=0,
                   two_sided_test=True,
                   within_grp=True,
                   random_state=None,
                   use_z=False,
                   demean_confounds=True,
                   min_vg_size=2)

    # All close, but not equal
    assert not np.array_equal(original_scores1, original_scores2)

def test_permuted_v_rand_vals_min_vg_size2():

    tested_vars=np.random.random((6))
    target_vars=np.random.random((6, 5))
    confounding_vars=np.random.random((6, 2))
    permutation_structure=np.array([1, 1, 1, 3, 3, 2])

    _, original_scores1, _ =\
        permuted_v(tested_vars=tested_vars,
                   target_vars=target_vars,
                   confounding_vars=confounding_vars,
                   permutation_structure=permutation_structure,
                   n_perm=0,
                   two_sided_test=True,
                   within_grp=True,
                   random_state=None,
                   use_tf=False,
                   use_z=False,
                   demean_confounds=True,
                   min_vg_size=2)

    # Should get same result from manual drop
    tested_vars = tested_vars[:-1]
    target_vars = target_vars[:-1]
    confounding_vars = confounding_vars[:-1]
    permutation_structure  = permutation_structure[:-1]

    _, original_scores2, _ =\
    permuted_v(tested_vars=tested_vars,
                target_vars=target_vars,
                confounding_vars=confounding_vars,
                permutation_structure=permutation_structure,
                n_perm=0,
                two_sided_test=True,
                within_grp=True,
                random_state=None,
                use_z=False,
                demean_confounds=True,
                min_vg_size=None)

    # All close, but not equal
    assert np.array_equal(original_scores1, original_scores2)

def test_permuted_v_rand_vals_no_covars():

    pvals, original_scores, h0_vmax =\
        permuted_v(tested_vars=np.random.random((4)),
                   target_vars=np.random.random((4, 5)),
                   confounding_vars=None,
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

def test_permuted_v_rand_vals_no_covars_model_intercept():

    pvals, original_scores, h0_vmax =\
        permuted_v(tested_vars=np.random.random((4)),
                   target_vars=np.random.random((4, 5)),
                   confounding_vars=None,
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
                   model_intercept=True,
                   demean_confounds=True,
                   min_vg_size=None)

    assert len(pvals) == 5
    assert len(original_scores) ==  5
    assert len(h0_vmax) == 10
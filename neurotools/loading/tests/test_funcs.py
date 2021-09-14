import numpy as np
from ..funcs import load

def test_load_basic():

    true = np.ones(10)
    assert np.array_equal(load('test_data/ones.npy'), true)
    assert np.array_equal(load('test_data/ones.nii'), true)
    assert np.array_equal(load('test_data/ones.nii.gz'), true)







import numpy as np
import os
from ..funcs import load

file_dr = os.path.dirname(os.path.realpath(__file__))

def test_load_basic():

    true = np.ones(10)
    assert np.array_equal(load(os.path.join(file_dr, 'test_data/ones.npy')), true)
    assert np.array_equal(load(os.path.join(file_dr, 'test_data/ones.nii')), true)
    assert np.array_equal(load(os.path.join(file_dr, 'test_data/ones.nii.gz')), true)







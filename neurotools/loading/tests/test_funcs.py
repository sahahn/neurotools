import numpy as np
import os
from ..funcs import load
import nibabel as nib
import shutil

file_dr = os.path.dirname(os.path.realpath(__file__))

def test_load_basic():

    os.makedirs(os.path.join(file_dr, 'test_data'), exist_ok=True)

    x = np.ones(10)
    p1 = os.path.join(file_dr, 'test_data', 'ones.npy')
   
    np.save(p1, x)

    assert np.array_equal(load(p1), x)

    p2 = os.path.join(file_dr, 'test_data', 'ones.nii')
    p3 = os.path.join(file_dr, 'test_data', 'ones.nii.gz')
    x_nib = nib.Nifti1Image(x, np.eye(4))
    nib.save(x_nib, p2)
    nib.save(x_nib, p3)

    assert np.array_equal(load(p2), x)
    assert np.array_equal(load(p3), x)

    shutil.rmtree(os.path.join(file_dr, 'test_data'))




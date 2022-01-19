from ..space import process_space
import numpy as np

def test_basic_arr_inputs():

    # Fsaverage 3, one hemi, needs medial test case
    proc_data, space = process_space(np.zeros(578), hemi='lh')
    assert proc_data['lh'].shape == (642, )
    assert 'rh' not in proc_data
    assert 'vol' not in proc_data
    assert 'sub' not in proc_data
    assert space == 'fsaverage3'

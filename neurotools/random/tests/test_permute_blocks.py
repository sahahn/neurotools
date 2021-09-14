import numpy as np
from ..permute_blocks import block_permutation

def test_basic_example():

    blocks = np.array([[-1, 1, -1, 1],
                       [-1, 1, -1, 1],
                       [-1, 1, -2, 1],
                       [-1, 1, -2, 1],
                       [-1, 2, -3, 2],
                       [-1, 2, -3, 2],
                       [-1, 2, -3, 1],
                       [-1, 2, -4, 2],
                       [-1, 2, -4, 1],
                       [-1, 2, -4, 2],
                       [-1, 3, -5, 3],
                       [-1, 3, -6, 3],
                       [-1, 3, -7, 3]])

    indx = np.arange(blocks.shape[0])
    
    # Permute randomly 500 times, testing
    # different constraints.

    spot0 = []
    for random_state in range(500):

        new = block_permutation(x=indx, blocks=blocks, random_state=random_state)
        spot0.append(new[0])
        
        # Make sure main blocks didn't swap
        assert 0 in new[:4]
        assert 1 in new[:4]
        assert 2 in new[:4]
        assert 3 in new[:4]
        assert 4 in new[4:10]
        assert 5 in new[4:10]
        assert 6 in new[4:10]
        assert 7 in new[4:10]
        assert 8 in new[4:10]
        assert 9 in new[4:10]
        assert 10 in new[10:]
        assert 11 in new[10:]
        assert 12 in new[10:]

        # These can only swap with each other
        assert new[4] in [4, 5, 7, 9]
        assert new[5] in [4, 5, 7, 9]
        assert new[7] in [4, 5, 7, 9]
        assert new[9] in [4, 5, 7, 9]
        assert new[6] in [6, 8]
        assert new[8] in [6, 8]
    
    # First index should get all of these
    assert 0 in spot0
    assert 1 in spot0
    assert 2 in spot0
    assert 3 in spot0




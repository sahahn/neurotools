import numpy as np
from ..permute_blocks import block_permutation, get_auto_vg

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

def assert_vgs_equal(vg1, vg2):
    '''Comparison between two vgs,
    make sure they are functionally the same'''
    
    # Same number of unique vgs
    assert len(np.unique(vg1)) == len(np.unique(vg2))
    
    # Go through each set of unique groups from vg1
    for u in np.unique(vg1):
        u_inds = np.where(vg1==u)
        
        # Make sure at the same spot in the second
        # vg, it has all of the same number
        assert len(np.unique(vg2[u_inds]))

def test_auto_vg1():

    ex_block = np.array([[-1, 1, 1],
                        [-1, 1, 2],
                        [-1, 2, 1],
                        [-1, 2, 2],
                        [-1, 3, 1],
                        [-1, 3, 2]])

    true_vg = np.array([1, 1, 2, 2, 3, 3])
    auto_vg = get_auto_vg(ex_block)
    assert_vgs_equal(true_vg, auto_vg)

def test_auto_vg2():
    '''More complex example from permute test'''

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


    true_vg = np.array([1, 1, 1, 1, 2, 2, 3, 2, 3, 2, 4, 4, 4])
    auto_vg = get_auto_vg(blocks)
    assert_vgs_equal(true_vg, auto_vg)


def test_auto_vg3():

    ex_block = np.array([[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 2],
                         [1, 1, 2],
                         [1, 2, 1],
                         [1, 2, 1],
                         [1, 2, 2],
                         [1, 2, 2]])

    true_vg = np.array([1, 2, 1, 2, 1, 2, 1, 2])
    auto_vg = get_auto_vg(ex_block)
    assert_vgs_equal(true_vg, auto_vg)

def test_auto_vg4():

    ex_block = np.array([[1, 1],
                         [1, 1],
                         [1, 1],
                         [1, 2],
                         [1, 2],
                         [1, 2]])

    true_vg = np.array([1, 2, 3, 1, 2, 3])
    auto_vg = get_auto_vg(ex_block)
    assert_vgs_equal(true_vg, auto_vg)



def test_auto_vg5():

    ex_block = np.array([[1, 1],
                         [1, 1],
                         [1, 1],
                         [1, 2],
                         [1, 2],
                         [1, 2],
                         [1, 3],
                         [1, 3],
                         [1, 3]])

    true_vg = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
    auto_vg = get_auto_vg(ex_block)
    assert_vgs_equal(true_vg, auto_vg)


def test_auto_vg6():
    
    # Default case is to assume the first
    # column is within block shuffling
    ex_block = np.array([[1],
                         [1],
                         [1],
                         [2],
                         [2],
                         [2]])

    true_vg = np.array([1, 1, 1, 2, 2, 2])
    auto_vg = get_auto_vg(ex_block)
    assert_vgs_equal(true_vg, auto_vg)


def test_auto_vg7():

    ex_block = np.array([[-1, 1],
                         [-1, 1],
                         [-1, 1],
                         [-1, 2],
                         [-1, 2],
                         [-1, 2]])

    true_vg = np.array([1, 1, 1, 2, 2, 2])

    auto_vg = get_auto_vg(ex_block)
    assert_vgs_equal(true_vg, auto_vg)

def test_auto_vg7():

    ex_block = np.array([[-1, 2],
                         [-1, 2],
                         [-1, 2],
                         [-1, 1],
                         [-1, 1],
                         [-1, 1]])

    true_vg = np.array([1, 1, 1, 2, 2, 2])

    auto_vg = get_auto_vg(ex_block)
    assert_vgs_equal(true_vg, auto_vg)

def test_auto_vg8():

    ex_block = np.array([[-1, 1],
                         [-1, 2],
                         [-1, 1],
                         [-1, 2],
                         [-1, 1],
                         [-1, 2]])

    true_vg = np.array([1, 2, 1, 2, 1, 2])
    auto_vg = get_auto_vg(ex_block)
    assert_vgs_equal(true_vg, auto_vg)


def test_auto_vg9_1D_input():
    '''Should handle passed as 1D array case'''

    ex_block = np.array([1, 1, 1, 2, 2, 2])
    true_vg = np.array([1, 1, 1, 2, 2, 2])
    auto_vg = get_auto_vg(ex_block)
    assert_vgs_equal(true_vg, auto_vg)

def test_auto_vg_10():

    # Is this right?
    ex_block = np.array([[-1, 1, 1],
                         [-1, 1, 1],
                         [-1, 1, 2],
                         [-1, 1, 2],
                         [-1, 2, 1],
                         [-1, 2, 2]])

    true_vg = np.array([1, 2, 1, 2, 3, 3])
    auto_vg = get_auto_vg(ex_block)
    assert_vgs_equal(true_vg, auto_vg)

def test_auto_vg_11():
    ex_block = np.array([[1, 1],
                         [1, 2],
                         [1, 3],
                         [1, 4],
                         [1, 5],
                         [1, 6]])

    true_vg = np.array([1, 1, 1, 1, 1, 1])
    auto_vg = get_auto_vg(ex_block)
    assert_vgs_equal(true_vg, auto_vg)

def test_auto_vg_12():
    ex_block = np.array([[1, 1],
                         [1, 1],
                         [1, 2],
                         [1, 2],
                         [1, 3],
                         [1, 3]])

    true_vg = np.array([1, 2, 1, 2, 1, 2])
    auto_vg = get_auto_vg(ex_block)
    assert_vgs_equal(true_vg, auto_vg)

def test_auto_vg_13():
    ex_block = np.array([[1, -1, 1],
                         [1, -1, 2],
                         [1, -2, 1],
                         [1, -2, 2],
                         [1, -3, 1],
                         [1, -3, 2]])

    true_vg = np.array([1, 2, 1, 2, 1, 2])
    auto_vg = get_auto_vg(ex_block)
    assert_vgs_equal(true_vg, auto_vg)

def test_auto_vg_14():
    ex_block = np.array([[1, -1, 1],
                         [1, -1, 1],
                         [1, -1, 2],
                         [1, -2, 1],
                         [1, -2, 1],
                         [1, -2, 2],
                         [1, -3, 1],
                         [1, -3, 2]])

    true_vg = np.array([1, 1, 2, 1, 1, 2, 3, 4])
    auto_vg = get_auto_vg(ex_block)
    assert_vgs_equal(true_vg, auto_vg)

def test_auto_vg_15():
    ex_block = np.array([[1, -1, 1],
                         [1, -1, 1],
                         [1, -1, 2],
                         [1, -2, 1],
                         [1, -2, 1],
                         [1, -2, 2],
                         [1, -3, 1],
                         [1, -3, 1]])

    true_vg = np.array([1, 1, 2, 1, 1, 2, 3, 3])
    auto_vg = get_auto_vg(ex_block)
    assert_vgs_equal(true_vg, auto_vg)
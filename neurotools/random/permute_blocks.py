import numpy as np
import pandas as pd
from sklearn.utils import check_random_state


def _within_grp_permute(blocks, rng):
    '''If here, means performing permutations within each
    unique group. By default this should be called first?'''
    
    # If reached the index layer - stop
    if blocks.shape[1] == 1:
        return blocks
    
    # Get all unique groups
    col = blocks[:, 0]
    unique_grps = np.unique(col)
    
    # Process each group separate
    for grp in unique_grps:
        
        # Get index of members of this group
        grp_inds = np.where(col == grp)[0]
        
        # Neg case, within grp permute
        if grp < 0:
            blocks[grp_inds, 1:] = _within_grp_permute(blocks[grp_inds, 1:], rng)
        
        # Pos case is swapping groups
        else:
            blocks[grp_inds, 1:] = _between_grp_permute(blocks[grp_inds, 1:], rng)
            
    return blocks
            

def _between_grp_permute(blocks, rng):
    '''Permute between groups of the same size'''
    
    # Get all unique groups
    col = blocks[:, 0]
    unique_grps, cnts = np.unique(col, return_counts=True)
    
    # Fill in new_blocks with permuted copy
    new_blocks = blocks.copy()
    
    # For each unique size of group, permute with other groups
    for grp_size in np.unique(cnts):
        
        # Get a permuted order for all valid groups
        grps_of_size = unique_grps[np.where(cnts==grp_size)]
        permuted_grps = rng.permutation(grps_of_size)
        
        # Apply the permuted order
        original_inds = [np.where(col == grp) for grp in grps_of_size]
        new_inds = [np.where(col == grp) for grp in permuted_grps]
        
        for o_inds, n_inds in zip(original_inds, new_inds):
            new_blocks[n_inds] = blocks[o_inds]
            
    # Still need to apply within grp - if any, to next level of data
    # Return the results of that operation
    return _within_grp_permute(new_blocks, rng)

def permute_blocks(in_blocks, rng):
    '''Function to perform permutations according to a PALM style
    Exchangeability Block structure.

    The sorting by lex order, then reversing is
    to make sure inter block swaps are done in a valid way according
    to any subtypes
    '''
    
    # Add index as column to blocks - keeping track of permutations
    indx = np.expand_dims(np.arange(in_blocks.shape[0]), 1)
    blocks = np.hstack([in_blocks, indx])

    # Generate lexsort order from left to right,
    # from in_blocks and apply
    lex_sort_order = np.lexsort([in_blocks[:, -i] for i in range(in_blocks.shape[1])])
    blocks = blocks[lex_sort_order]
    
    # Start always as within grp on the outer layer of exchange blocks
    permuted_blocks = _within_grp_permute(blocks, rng)

    # Reverse initial lex sort
    rev_lex_sort_order =\
        [np.where(lex_sort_order == i)[0][0] for i in range(len(lex_sort_order))]
    permuted_blocks = permuted_blocks[rev_lex_sort_order]
    
    # Only need to return the new permuted index
    return permuted_blocks[:, -1]

def block_permutation(x, blocks, random_state=None):
    '''
    Randomly permute a sequence, or return a permuted range.
    If x is a multi-dimensional array, it is only shuffled along its first index.
    Performs permutations according to passed blocks, where blocks
    are specified with PALM style Exchangeability Blocks.
    '''
    
    # Make sure blocks are np array
    if isinstance(blocks, pd.DataFrame):
        blocks = np.array(blocks)
    
    # Proc. random state
    rng = check_random_state(random_state)

    # Permute blocks one
    permuted_indx = permute_blocks(blocks, rng)
    
    # Return original sequence in new permuted order
    return x[permuted_indx]
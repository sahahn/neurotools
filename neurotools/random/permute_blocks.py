import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder


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

def _proc_block_input(in_blocks, within_grp=True):

    # If passed as df, convert to array
    if isinstance(in_blocks, (pd.DataFrame, pd.Series)):
        in_blocks = np.array(in_blocks)

    # Passed 1D array case - expand
    if len(in_blocks.shape) == 1:
        in_blocks = np.expand_dims(in_blocks, axis=1)

    # Get outer col to add to left side
    ones = np.ones((len(in_blocks), 1), dtype='int')
    if within_grp:
        ones *= -1
    
    # Add index as column to blocks - to keep track of permutations
    indx = np.expand_dims(np.arange(in_blocks.shape[0], dtype='int'), 1)

    # Add index and ones cols and make sure blocks are int
    blocks = np.hstack([ones, in_blocks, indx]).astype('int')

    # Generate lexsort order from left to right,
    # from in_blocks and apply. Basically we need to apply this
    # ordering in order to ensure that any whole-block shuffling that occurs,
    # occurs such that the order is preserved.
    lex_sort_order = np.lexsort([in_blocks[:, -i] for i in range(in_blocks.shape[1])])
    blocks = blocks[lex_sort_order]

    # Reverse initial lex sort, and return that along with the blocks
    rev_lex_sort_order =\
        [np.where(lex_sort_order == i)[0][0] for i in range(len(lex_sort_order))]

    return blocks, rev_lex_sort_order


def permute_blocks(in_blocks, rng, within_grp=True):
    '''Function to perform permutations according to a PALM style
    Exchangeability Block structure.

    The sorting by lex order, then reversing is
    to make sure inter block swaps are done in a valid way according
    to any subtypes
    '''

    # Prep blocks for permutations - adding ones and index cols
    # Then sorting, such that we can make assumptions
    # about the ordering of whole block swaps
    blocks, rev_lex_sort_order = _proc_block_input(in_blocks, within_grp=within_grp)

    # Start always as within grp on the outer layer of exchange blocks
    # the within_grp arg controls the behavior here.
    permuted_blocks = _within_grp_permute(blocks, rng)

    # Apply the reverse of the lexical sorting
    permuted_blocks = permuted_blocks[rev_lex_sort_order]
    
    # Only need to return the new permuted index
    return permuted_blocks[:, -1]

def block_permutation(x, blocks, random_state=None, within_grp=True):
    '''
    Randomly permute a sequence, or return a permuted range.
    If x is a multi-dimensional array, it is only shuffled along its first index.
    Performs permutations according to passed blocks, where blocks
    are specified with PALM style Exchangeability Blocks.
    '''
    
    # Proc. random state
    rng = check_random_state(random_state)

    # Permute blocks one
    permuted_indx = permute_blocks(blocks, rng, within_grp=within_grp)

    # TODO fix this case
    if isinstance(x, (pd.Series, pd.DataFrame)):
        pass
    
    # Return original sequence in new permuted order
    return x[permuted_indx]


def get_auto_vg(in_blocks, within_grp=True):
    '''The idea with the variance groups is that
    any subjects which can swap with each other should
    be in the same variance group
    but any that cannot, should be in different ones.'''

    # Prep blocks for determine vg
    blocks, _ = _proc_block_input(in_blocks, within_grp=within_grp)
    
    # Init vg's as empty array
    vgs = np.zeros(len(blocks), dtype='int')
    can_swap = {i: set() for i in range(len(blocks))}
    
    def set_vg_val(ind, vg):
        
        # If ind is array-like
        if hasattr(ind, '__iter__'):
            
            # If both are array-like
            if hasattr(vg, '__iter__'):
                for i, v in zip(ind, vg):
                    set_vg_val(i, v)
            
            # If just ind is array like
            else:
                for i in ind:
                    set_vg_val(i, vg)
        
        # Set single value case
        else:
            
            # When trying to set a new
            # value, we need to check
            # to see if the ind being
            # set has already been set by
            # one of its can swaps
            # if it has be set by a can swap
            # use that value instead of 
            # the one being set
            for s_i in can_swap[ind]:
                if vgs[s_i] != 0:
                    vgs[ind] = vgs[s_i]
                    return
            
            # Otherwise, set as is
            vgs[ind] = vg

    def within_grp_case(blocks, vg):
        
        # Get current column
        col = blocks[:, 0]

        # If at final index layer, set to passed vg
        if blocks.shape[1] == 1:
            set_vg_val(col, vg)
            return 

        # Get all unique groups
        unique_grps = np.unique(col)

        # For each group in the within group
        # case, the idea is that they should
        # get a new unique vg
        for grp in unique_grps:

            # Get index of members of this group
            grp_inds = np.where(col == grp)[0]
            
            # Get next highest vg
            new_vg = np.max(vgs) + 1

            # Neg, within group case
            if grp < 0:
                within_grp_case(blocks[grp_inds, 1:], vg=new_vg)

            # Pos case is swapping groups
            else:
                between_grp_case(blocks[grp_inds, 1:], vg=new_vg)

        return blocks
    
    def between_grp_case(blocks, vg):
        
        # Get current column
        col = blocks[:, 0]

        # If at final index layer, set to passed vg
        if blocks.shape[1] == 1:
            set_vg_val(col, vg)
            return 
        
        # Okay for the between grp case, what
        # we want to do is assign the same variance
        # group sequentially to all groups of the same size.
        
        # Get all unique groups and their sizes
        unique_grps, cnts = np.unique(col, return_counts=True)
        
        # For each unique size of group, keep track of
        # unique in pass along
        cnt, pass_along = vg, np.zeros(len(col))
        for grp_size in np.unique(cnts):

            # Get a permuted order for all valid groups
            grps_of_size = unique_grps[np.where(cnts==grp_size)]
            grp_inds = np.array([np.where(col == grp)[0] for grp in grps_of_size])
            
            # Set each sequentially
            for inds in grp_inds.T:
                pass_along[inds] = cnt
                cnt += 1
                
        # Get next_col
        next_col = blocks[:, 1]
                
        # If next layer is the index, end case
        if blocks.shape[1] == 2:
            set_vg_val(next_col, pass_along)
            return
        
        # Get unique groups as combo between columns
        temp_col = np.stack([next_col, pass_along], axis=1)
        unique_grps = np.unique(temp_col, axis=0)
        
        # Proc each unique combo group simmilar to within_grp_case
        # Basically the idea is that, every member
        # in the same group here should be in the same
        # variance group as they can swap.
        # We just need to consider the extra cases
        # where based on the values of the next col,
        # there might be further overlapping groups
        for grp in unique_grps:
            
            # Get grp inds as binary array
            grp_inds = np.all(temp_col == grp, axis=1)
            
            swaps = set(blocks[grp_inds, -1])
            for i in swaps:
                can_swap[i] = swaps
                
        # After values are set in can swap,
        # we can process the groups as is with
        # within group case - doesn't matter what vg val is passed
        within_grp_case(blocks, vg=None)
                
    # Start recursive loop with vg=0 within group case
    within_grp_case(blocks, vg=0)

    # Return vgs so that that they
    # got neatly from 0 to n-1
    return LabelEncoder().fit_transform(vgs)
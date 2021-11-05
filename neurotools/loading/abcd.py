import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from joblib.hashing import hash as joblib_hash
from sklearn.preprocessing import LabelEncoder
import os
from pathlib import Path

from ..misc.text import readable_size_to_bytes
from ..misc.print import _get_print
from .. import data_dr

def _base_load_from_csv(cols, csv_loc, eventname):
    '''The actual loading piece'''

    # Load from csv - using abcd specific NaN values
    data = pd.read_csv(csv_loc,
                       usecols=['src_subject_id', 'eventname'] + cols,
                       na_values=['777', 999, '999', 777])

    # If None, keep all and return
    if eventname is None:
        return data

    # Otherwise, we assume eventname is list
    data = data.loc[data[data['eventname'].isin(eventname)].index]

    # If just one eventname, drop the eventname col
    if len(eventname) == 1:
        data = data.drop('eventname', axis=1)

    return data

def _get_cache_dr(cache_dr):

    # Process default
    if cache_dr == 'default':
        cache_dr = os.path.join(data_dr, 'abcd_csv_cache')

    # Give error if exists and not directory
    if os.path.exists(cache_dr) and not os.path.isdir(cache_dr):
        raise RuntimeError(f'Passed cache_dr: {cache_dr} already exists and is not a directory!')

    # Make sure exists if doesn't exist
    os.makedirs(cache_dr, exist_ok=True)

    return cache_dr

def _get_load_hash_str(cols, csv_loc, eventname):

    # Sort cols for same behavior each time
    sorted_cols = sorted(cols)

    # Same with eventname, but if None, keep as None
    try:
        sorted_eventname = sorted(eventname)
    except TypeError:
        sorted_eventname = None
    
    # Use base name for csv instead of absolute path
    csv_base_name = os.path.basename(csv_loc)

    # Return joblib hash
    return joblib_hash([sorted_cols, csv_base_name, sorted_eventname])

def _get_load_cache_loc(cols, csv_loc, eventname, cache_dr):

    # Otherwise, caching behavior is expected
    # Get unique hash for these arguments - with tolerance to order and
    # and a few other things.
    hash_str = _get_load_hash_str(cols, csv_loc, eventname)

    # Determine loc from hash_str
    cache_loc = os.path.join(cache_dr, hash_str)

    return cache_loc

def _order_cache_loaded(data, cols):
    '''When loading from hash we make sure to return as ordered
    by cols in the cases where the saved copy was saved with a different
    column order.'''

    # Generate correct ordering
    ordered_cols = ['src_subject_id']

    # Eventname may or may not be there
    if 'eventname' in data:
        ordered_cols += ['eventname']

    # Add rest of cols
    ordered_cols += cols

    # Apply the ordering and return
    return data[ordered_cols]

def _check_cache_sz_limit(cache_dr, cache_max_sz, _print):

    _print('Checking cache size limit', level=2)

    # Get full file paths of all cached
    all_cached = [os.path.join(cache_dr, f)
                  for f in os.listdir(cache_dr)]

    # Get size of current directory
    size = sum(os.path.getsize(f) for f in all_cached)

    # Make sure cache_max_sz as bytes
    cache_max_sz = readable_size_to_bytes(cache_max_sz)

    _print(f'Current saved size is: {size} our of passed cache_max_sz: {cache_max_sz}', level=1)

    # If over the current limit
    if size > cache_max_sz:

        # Sort all cached files from oldest to newest, based
        # on last modified.
        old_to_new = sorted(all_cached, key=os.path.getctime)

        # Delete in order from old to new
        # until under the limit
        removed, n = 0, 0
        while size - removed > cache_max_sz:

            # Select next to remove
            to_remove = old_to_new[n]
            
            # Update counters
            n += 1
            removed += os.path.getsize(to_remove)

            # Remove cached file
            os.remove(to_remove)

            _print(f'Removed cached file at: {to_remove}', level=2)


def _base_cache_load_from_csv(cols, csv_loc, eventname,
                              cache_dr, cache_max_sz, _print):

    # If no caching, base case, just load as normal
    if cache_dr is None:
        _print(f'No cache_dr specified, loading from {csv_loc}', level=1)
        return _base_load_from_csv(cols, csv_loc, eventname)

    # Make sure cache_dr arg
    cache_dr = _get_cache_dr(cache_dr)

    # Get cache loc
    cache_loc = _get_load_cache_loc(cols, csv_loc, eventname, cache_dr)

    # If exists, load from saved
    if os.path.exists(cache_loc):
        _print(f'Loading from cache_loc: {cache_loc}', level=1)

        # Cached data saved as tab seperated, without any index cols
        data = pd.read_csv(cache_loc, sep='\t')

        # Make sure to updating last modified of file to now
        Path(cache_loc).touch()

        # Set to correct column order
        data = _order_cache_loaded(data, cols)

    # If doesn't yet exist, load like normal
    else:
        _print(f'No existing cache found, loading from: {csv_loc}', level=1)

        # Base load
        data = _base_load_from_csv(cols, csv_loc, eventname)

        # Save a copy under the cache_loc as tsv, no index
        data.to_csv(cache_loc, index=False, sep='\t')

        _print(f'Saving loaded data to cache_loc: {cache_loc}', level=1)

    # Before returning - check the cache_max_sz argument
    # clearing any files over the limit
    _check_cache_sz_limit(cache_dr, cache_max_sz, _print=_print)

    return data

def load_from_csv(cols, csv_loc,
                  eventname='baseline_year_1_arm_1',
                  drop_nan=False, encode_cat_as='ordinal',
                  cache_dr='default', cache_max_sz='30G',
                  verbose=0):
    '''Special ABCD Study specific helper utility to load specific
    columns from a csv saved version of the DEAP release RDS
    file or simmilar ABCD specific csv dataset.

    Parameters
    ----------
    cols : str or list-like
        Either a single str with the column name to load,
        or a list / list-like with the names of multiple columns to
        load.

        If any variable passed is wrapped in 'C(variable_name)'
        that variable will be ordinalized (or whatever option is specified
        with the encode_cat_as option) and saved under the base variable name
        (i.e., with C() wrapper removed).

    csv_loc : str
        The str location of the csv saved version of the DEAP release
        RDS file for the ABCD Study - or any other comma seperated dataset
        with an eventname column.

    eventname : str, list-like or None, optional
        The single eventname as a str, or multiple eventnames
        in which to include results by. If passed as None then
        all avaliable data will be kept.

        If a single eventname is specified then the eventname
        column will be dropped, otherwise it will be kept.

        ::

            default = 'baseline_year_1_arm_1'

    drop_nan : bool, optional
        If True, then drop any rows / subjects data with
        missing values in any of the requested columns.

        Note: Any values encoded as ['777', 999, '999', 777]
        will be treated as NaN. This is a special ABCD specific consideration.

        ::

            default = False

    encode_cat_as : {'ordinal', 'one hot', 'dummy'}, optional
        The way in which categorical vars, any wrapped in C(),
        should be categorically encoded.

        - 'ordinal':
            The variables in encoded sequentially in one
            column with the original name, with values 0 to k-1
            where k is the number of unique categorical values.
            This method uses :class:`OrdinalEncoder<sklearn.preprocessing.OrdinalEncoder>`.

        - 'one hot':
            The variables is one hot encoded, adding columns
            for each unique value. This method uses function :func:`pandas.get_dummies`.

        - 'dummy':
            Same as 'one hot', except one of the columns is then dropped.

    Returns
    -------
    df : :class:`pandas.DataFrame`
        Will return a :class:`pandas.DataFrame` as indexed by
        column `src_subject_id` within the original csv.

    '''

    # Get verbose printer
    _print = _get_print(verbose)
    
    # Handle passing input as single str / column
    if isinstance(cols, str):
        cols = [cols]
    if isinstance(eventname, str):
        eventname = [eventname]

    # Make sure proper input for cols and eventname
    # if not list, try to cast to list in case of other list-like
    if not isinstance(cols, list):
        try:
            cols = list(cols)
        except:
            raise RuntimeError(f'Passed cols: {cols} must be passed as str or list-like!')

    # Extra is None case for eventname
    if not isinstance(eventname, list) and eventname is not None:
        try:
            eventname = list(eventname)
        except:
            raise RuntimeError(f'Passed eventname: {eventname} must be passed as str, list-like or None!')

    # Check for any wrapped in C()
    use_cols = cols
    cat_vars = []
    for i, col in enumerate(use_cols):
        if col.startswith('C(') and col.endswith(')'):
            var_name = col[2:-1]

            # Replace with unwrapped
            use_cols[i] = var_name

            # Keep track
            cat_vars.append(var_name)
    
    # Load with optional caching - only need
    # to cache this main operation.
    data = _base_cache_load_from_csv(
        cols=use_cols, csv_loc=csv_loc, eventname=eventname,
        cache_dr=cache_dr, cache_max_sz=cache_max_sz, _print=_print)

    # Set index as subject id
    data = data.set_index('src_subject_id')

    # Optionally drop NaN's
    if drop_nan:
        original_len = len(data)
        data.dropna(inplace=True)
        _print(f'Dropping {original_len - len(data)} subjects for missing data.', level=1)
    
    # Ordinally encode any C wrapped vars - categorical vars
    _print(f'Categorical vars to be encoded =', cat_vars, level=3)
    if len(cat_vars) > 0:

        _print(f'Encoding {len(cat_vars)} categorical vars with {encode_cat_as} encoding.', level=1)

        if encode_cat_as == 'ordinal':
            data[cat_vars] = OrdinalEncoder().fit_transform(data[cat_vars])
        elif encode_cat_as == 'one hot':
            data = pd.get_dummies(data=data, columns=cat_vars, drop_first=False)
        elif encode_cat_as == 'dummy':
            data = pd.get_dummies(data=data, columns=cat_vars, drop_first=True)
        else:
            raise RuntimeError(f'encode_cat_as {encode_cat_as} not valid option.')    

    return data


def load_family_block_structure(csv_loc, subjects=None,
                                eventname='baseline_year_1_arm_1',
                                add_neg_ones=False, cache_dr='default',
                                cache_max_sz='30G', verbose=0):
    '''This helper utility loads PALM-style exchanability blocks for ABCD study specific data
    according to right now a fixed set of rules:

    - Families of the same type can be shuffled (i.e., same number of members + of same status)
    - Siblings of the same type can be shuffled 
    - Treat DZ as ordinary sibs (i.e., just treat MZ seperately)

    Parameters
    ----------
    csv_loc : str / file path
        The location of the csv saved version of the DEAP release
        RDS file for the ABCD Study. This can also just be any
        other csv as long as it has columns:
        'rel_family_id', 'rel_relationship', 'genetic_zygosity_status_1'

    subjects : None or array-like, optional
        Can optionally specify that the block structure be created on a subset of subjects,
        though if any missing values are present in rel_relationship or rel_family_id
        within this subset, then those will be further dropped.

        If passed as non-null, this should be a valid array-like
        or :class:`pandas.Index` style set of subjects.

        ::

            default = None

    eventname : str, array-like or None, optional
        A single eventname as a str in which to specify data by.

        For now, this method only supports loading data at a single
        time point across subjects.

        ::

            default = 'baseline_year_1_arm_1'

    add_neg_ones : bool, optional
        If True, add a left-most column
        with all negative ones representing
        that swaps should occur within group at
        the outermost level. Note that if using
        a permutation function through neurotools
        that accepts this style of blocks,
        this outer layer is assumed by default,
        so this parameter can be left as False.

        ::

            default = False

    Returns
    --------
    block_structure : :class:`pandas.DataFrame`
        The loaded block structure as indexed by src_subject_id with
        columns: 'neg_ones', 'family_type', 'rel_family_id' and 'rel_relationship'.
        
        Any subjects with missing values in a key column have been dropped from
        this returned structure.
    '''

    # Get verbose printer
    _print = _get_print(verbose)

    # TODO: Add in support for what a block structure with
    # multiple eventnames, i.e., longitudinal data, and what
    #  that would look like.

    # Load in needed columns, w/o NaNs
    rel_cols = ['rel_family_id', 'rel_relationship', 'genetic_zygosity_status_1']
    data = load_from_csv(rel_cols, csv_loc,
                         eventname=eventname,
                         drop_nan=False,
                         cache_dr=cache_dr,
                         cache_max_sz=cache_max_sz,
                         verbose=verbose)
    _print(f'Loaded data needed with shape: {data.shape}', level=1)

    # Set to subset of passed subjects if any,
    if subjects is not None:
        data = data.loc[subjects]

    # Drop any data points if rel_family_id or rel_relationship is missing
    to_drop = data[pd.isnull(data[['rel_relationship',
                                   'rel_family_id']]).any(axis=1)].index
    data = data.drop(to_drop)

    # Treat twins if dy zy as normal siblings, basically if not labelled mono, then normal sib
    treat_as_siblings = data[((data['rel_relationship'] == 'twin') | (data['rel_relationship'] == 'triplet')) &\
                              (data['genetic_zygosity_status_1'] != 'monozygotic')].index
    data.loc[treat_as_siblings, 'rel_relationship'] = 'sibling'

    # Create dict mapping subj to family
    families = {}
    for subj in data.index:
        rel_family_id = data.loc[subj, 'rel_family_id']
        family = data[data['rel_family_id'] == rel_family_id].index
        families[subj] = family

    # Init column
    data['family_type'] = 0
    hashes = {}
    cnt = 0

    # Fill in unique value for every type of family
    for subj in data.index:
        
        # Get unique hash
        h = joblib_hash(data.loc[families[subj], 'rel_relationship'].values)
        
        try:
            u = hashes[h]
        except KeyError:
            hashes[h] = cnt
            u = cnt
            cnt += 1
            
        data.loc[subj, 'family_type'] = u

    # Ordinally encode cols
    data['rel_relationship'] = LabelEncoder().fit_transform(data['rel_relationship'])
    data['rel_family_id'] = LabelEncoder().fit_transform(data['rel_family_id'])

    # Add outer col with -1's if requested
    if add_neg_ones:
        data['neg_ones'] = -1

        return data[['neg_ones', 'family_type', 'rel_family_id', 'rel_relationship']]

    # Default order w/o neg ones
    return data[['family_type', 'rel_family_id', 'rel_relationship']]

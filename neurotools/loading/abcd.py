import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from joblib.hashing import hash as joblib_hash
from sklearn.preprocessing import LabelEncoder
from ..misc.print import _get_print
from .cache import _base_cache_load


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

def _csv_cache_load_func(cache_loc, load_args):

    # Cached data saved as tab seperated, without any index cols
    data = pd.read_csv(cache_loc, sep='\t')

    # Set to correct column order
    data = _order_cache_loaded(data, load_args[0])

    return data

def _csv_cache_save_func(data, cache_loc):

     # Save a copy under the cache_loc as tsv, no index
    data.to_csv(cache_loc, index=False, sep='\t')

def load_from_csv(cols, csv_loc,
                  eventname='baseline_year_1_arm_1',
                  drop_nan=False, encode_cat_as='ordinal',
                  verbose=0, **cache_args):
    '''Special ABCD Study specific helper utility to load specific
    columns from a csv saved version of the DEAP release RDS
    file or similar ABCD specific csv dataset.

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

        - 'ordinal' :
          
          The variables in encoded sequentially in one
          column with the original name, with values 0 to k-1
          where k is the number of unique categorical values.
          This method uses :class:`OrdinalEncoder<sklearn.preprocessing.OrdinalEncoder>`.

        - 'one hot' :
          
          The variables is one hot encoded, adding columns
          for each unique value. This method uses function :func:`pandas.get_dummies`.

        - 'dummy' :
          
          Same as 'one hot', except one of the columns is then dropped.

    cache_args : keyword arguments
        There are a number of optional cache arguments that can be set via
        kwargs, as listed below.

        - **cache_dr** : str or None
        
          The location of where to cache the results of
          this function, for faster loading in the future.

          If None, do not cache. The default if not set is
          'default', which will use caching in a location defined
          by the function name in a folder called
          neurotools_cache in the users homes directory.

        - **cache_max_sz** : str or int
        
          This parameter defines the maximize size of
          the cache directory. The idea is that if saving a new
          cached function call and it exceeds this cache max size,
          previous saved caches (by oldest in terms of used) will be
          deleted, ensuring the cache directory remains under this size.

          Can either pass in terms of bytes directly as a number,
          or in terms of a str w/ byte marker, e.g., '14G' for
          14 gigabytes, or '10 KB' for 10 kilobytes.

          The default if not set is '30G'.

        - **use_base_name** : bool
            
          Optionally when any arguments used in the
          caching can be cached based on either their full file
          path, if use_base_name is False, or just the file name itself,
          so for example /some/path/location vs. just location.
          The default if not set is True, as it assumes that maybe
          another file in another location with the same name is the same.

    Returns
    -------
    df : :class:`pandas.DataFrame`
        Will return a :class:`pandas.DataFrame` as indexed by
        column `src_subject_id` within the original csv.

    '''

    # TODO move default cache location to
    # user's home directory ??? 

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
    use_cols = cols.copy()
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
    data = _base_cache_load(load_args=(use_cols, csv_loc, eventname),
                            load_func=_base_load_from_csv,
                            cache_load_func=_csv_cache_load_func,
                            cache_save_func=_csv_cache_save_func,
                            cache_args=cache_args,
                            _print=_print)

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
        _print(f'Set to {len(data)} subjects', level=1)

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

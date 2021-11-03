import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from joblib.hashing import hash as joblib_hash
from sklearn.preprocessing import LabelEncoder

# @TODO add option for caching

def load_from_csv(cols, csv_loc,
                  eventname='baseline_year_1_arm_1',
                  drop_nan=False, encode_cat_as='ordinal'):
    '''Special ABCD Study specific helper utility to load specific
    columns from a csv saved version of the DEAP release RDS
    file or simmilar ABCD specific csv dataset.

    Parameters
    ----------
    cols : str or array-like
        Either a single str with the column name to load,
        or a list / array-like with the names of multiple columns to
        load.

        If any variable passed is wrapped in 'C(variable_name)'
        that variable will be ordinalized (or whatever option is specified
        with the encode_cat_as option) and saved under the base variable name
        (i.e., with C() wrapper removed).

    csv_loc : str
        The str location of the csv saved version of the DEAP release
        RDS file for the ABCD Study - or any other comma seperated dataset
        with an eventname column.

    eventname : str, array-like or None, optional
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
    
    # Handle passing cols as single column
    if isinstance(cols, str):
        cols = [cols]

    # Check for any wrapped in C()
    cat_vars = []
    for i, col in enumerate(cols):
        if col.startswith('C(') and col.endswith(')'):
            var_name = col[2:-1]

            # Replace with unwrapped
            cols[i] = var_name

            # Keep track
            cat_vars.append(var_name)
    
    # Load from csv
    data = pd.read_csv(csv_loc,
                       usecols=['src_subject_id', 'eventname'] + cols,
                       na_values=['777', 999, '999', 777])
    
    # Load single eventname only
    if isinstance(eventname, str):
        data = data.loc[data[data['eventname'] == eventname].index]
        
        # Only drop eventname if a single eventname is kept
        data = data.drop('eventname', axis=1)
    
    # Index by multiple eventnames
    elif isinstance(eventname, list):
        data = data.loc[data[data['eventname'].isin(eventname)].index]

    # Set index
    data = data.set_index('src_subject_id')
    
    # Optionally drop NaN's
    if drop_nan:
        data.dropna(inplace=True)
    
    # Ordinally encode any C wrapped vars - categorical vars
    if len(cat_vars) > 0:

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
                                eventname='baseline_year_1_arm_1', add_neg_ones=False):
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

    # TODO: Add in support for what a block structure with
    # multiple eventnames, i.e., longitudinal data, and what
    #  that would look like.

    # Load in needed columns, w/o NaNs
    rel_cols = ['rel_family_id', 'rel_relationship', 'genetic_zygosity_status_1']
    data = load_from_csv(rel_cols, csv_loc,
                         eventname=eventname,
                         drop_nan=False)

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

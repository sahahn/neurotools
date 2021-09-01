from numpy.core.fromnumeric import var
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def load_from_abcd_rds(cols, rds_loc, eventname='baseline_year_1_arm_1', drop_nan=False):
    '''Special function to load specific columns from a csv saved
    version of the DEAP release RDS file for the ABCD Study.

    Note: special ABCD NaN indicators ['777', 999, '999', 777] will
    be treated as NaN.

    Parameters
    ----------
    cols : str or array-like
        Either a single str with the column name to load,
        or a list / array-like with the names of multiple columns to
        load.

        If any variable passed is wrapped in 'C(variable_name)'
        that variable will be ordinalized and saved under variable name.

    rds_loc : str / file path
        The location of the csv saved version of the DEAP release
        RDS file for the ABCD Study.

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

        ::

            default = False

    Returns
    -------
    df : pandas DataFrame
        Will return a pandas DataFrame as index'ed by src_subject_id, with
        the requested cols as columns.

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
    data = pd.read_csv(rds_loc,
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
    
    # Ordinally encode any C wrapped vars
    if len(cat_vars) > 0:
        data[cat_vars] = OrdinalEncoder().fit_transform(data[cat_vars])

    return data
import pandas as pd
import re

def name_replace(in_obj, to_replace, replace_with):
    '''Helper function to perform simple text replacement but
    over arbitrary objects.

    Parameters
    -----------
    in_obj : list, set or :class:`pandas.DataFrame`
        The object in which to replace all sub objects of,
        or in the case of DataFrame input the column names of.
        
        Note: This replacement occurs in some cases in-place rather
        than on a copy of the object (except in the case of set input).

    to_replace : str or list of str
        The str in which to replace with other argument replace_with.
        In the case that a list of str is passed, then
        any of the elements of the passed list will be if found replaced with
        the passed argument for replace_with.

    replace_with : str
        The str in which all instances of to_replace should be replaced with.

    Returns
    --------
    out_obj : list, set or :class:`pandas.DataFrame`
        The in_obj but with text replacement performed / column name replacement.


    Examples
    ----------

    .. ipython:: python

        import pandas as pd
        from neurotools.misc.text import name_replace
        
        # Base case
        name_replace(['cow123', 'goat123', 'somethingelse'], '123', '456')
        
        # DataFrame case
        df = pd.DataFrame(columns=['cow123', 'goat123', 'somethingelse'])
        name_replace(df, '123', '456')
        
        # to_replace list case
        name_replace(in_obj=['cow123', 'goat123', 'somethingelse'],
                     to_replace=['123', 'something'],
                     replace_with='456')

    '''
    
    # List case
    if isinstance(in_obj, list):
        for i in range(len(in_obj)):
            if isinstance(to_replace, list):
                for tr in to_replace:
                    in_obj[i] = in_obj[i].replace(tr, replace_with)
            else:
                in_obj[i] = in_obj[i].replace(to_replace, replace_with)

        return in_obj

    # Set case
    if isinstance(in_obj, set):

        new_obj = set()
        for item in in_obj:
            new_item = item

            if isinstance(to_replace, list):
                for tr in to_replace:
                    new_item = new_item.replace(tr, replace_with)
            else:
                new_item = new_item.replace(to_replace, replace_with)

            new_obj.add(new_item)

        return new_obj
    
    # DataFrame case
    if isinstance(in_obj, pd.DataFrame):

        cols = list(in_obj)
        col_name_replace = {}

        for col in cols:
            if isinstance(to_replace, list):
                for tr in to_replace:
                    if tr in col:
                        col_name_replace[col] = col.replace(tr, replace_with)
            else:
                if to_replace in col:
                    col_name_replace[col] = col.replace(to_replace, replace_with)

        return in_obj.rename(col_name_replace, axis=1)

    raise RuntimeError(f'type of {in_obj} not supported.')

def readable_size_to_bytes(size):
    '''Based on https://stackoverflow.com/questions/42865724/parse-human-readable-filesizes-into-bytes'''
    
    # Assume already bytes if fails convert to float
    try:
        return float(size)
    except ValueError:
        pass

    # Process as str, with 2 versions
    units = {"B": 1,
             "KB": 2**10, "K": 2**10,
             "MB": 2**20, "M": 2**20,
             "GB": 2**30, "G": 2**30,
             "TB": 2**40, "T": 2**40}
    
    # Make sure all uppercase
    size = size.upper()

    # Match replace with or w/o B
    if not re.match(r' ', size):
        size = re.sub(r'([KMGT]B?)', r' \1', size)

    # Split number and unit
    number, unit = [string.strip() for string in size.split()]

    # Return number * unit as int
    return int(float(number)*units[unit])

def substrs(x):
    return {x[i:i+j] for i in range(len(x)) for j in range(len(x) - i + 1)}

def find_substr(data):

    s = substrs(data[0])

    for val in data[1:]:
        s.intersection_update(substrs(val))

    try:
        mx = max(s, key=len)

    except ValueError:
        mx = ''

    return mx

def get_top_substrs(keys):

    found = []
    top = find_substr(keys)

    while len(top) > 1:
        found.append(top)

        keys = [k.replace(top, '') for k in keys]
        top = find_substr(keys)

    return found

def get_unique_str_markers(strs):

    # Get all top common sub strings
    top_substrs = get_top_substrs(strs)

    # Remove all common strs from each individual str
    unique_strs = []
    for s in strs:
        piece = s
        for substr in top_substrs:
            piece = piece.replace(substr, '')

        # Add final
        unique_strs.append(piece)

    return unique_strs
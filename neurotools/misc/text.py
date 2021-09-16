import pandas as pd

def name_replace(in_obj, to_replace='.', replace_with='-'):
    
    # List case
    if isinstance(in_obj, list):
        for i in range(len(in_obj)):
            if to_replace in in_obj[i]:
                in_obj[i] = in_obj[i].replace(to_replace, replace_with)

        return in_obj

    # Set case
    if isinstance(in_obj, set):

        new_obj = set()
        for item in in_obj:
            new_obj.add(item.replace(to_replace, replace_with))

        return new_obj
    
    # DataFrame case
    if isinstance(in_obj, pd.DataFrame):

        cols = list(in_obj)
        col_name_replace = {}

        for col in cols:
            if to_replace in col:
                col_name_replace[col] = col.replace(to_replace, replace_with)

        return in_obj.rename(col_name_replace, axis=1)

    raise RuntimeError(f'type of {in_obj} not supported.')

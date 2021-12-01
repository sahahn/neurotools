import numpy as np

def drop_top_x_outliers(data, subjects=None, top=50):
    '''Drop a fixed number of outliers from a set of loaded subjects data,
    based on the absolute mean of each subject's data.

    Parameters
    -----------
    data : numpy array
        Loaded 2D numpy array with shape # of subjects x data points,
        representing the loaded data in which to drop outliers from.

    subjects : numpy array or None, optional
        A corresponding list or array like of subjects with the same
        length as the first dimension of the passed data. If passed,
        then will return as a modified subject array representing
        the new subject list with dropped outliers excluded.
        If not passed, i.e., left as default None,
        will only return the modified data.

        ::

            default = None

    top : int, optional
        The number of subjects to drop.

        ::

            default = 50

    Returns
    --------
    data : numpy array
        The loaded 2D data, but with `top` subjects data removed.

    subjects : numpy array
        If passed originally as None, only data will be returned,
        otherwise this will represent the new list of kept subjects,
        with `top` removed.

    '''

    abs_data = np.abs(data)
    m = np.mean(abs_data, axis=1)
    to_drop = np.argsort(m)[-top:]
    to_keep = np.ones(m.shape, dtype='bool')
    to_keep[to_drop] = 0

    if subjects is not None:
        return data[to_keep], np.array(subjects)[to_keep]
    
    # Otherwise return just data
    return data[to_keep]
import warnings

def _get_print(verbose, _print=None):
    
    # Helper for muted print
    if verbose is None:
        return _get_print(verbose=-10)
    
    # If already set
    if not _print is None:
        return _print

    def _print(*args, **kwargs):

        if 'level' in kwargs:
            level = kwargs.pop('level')
        else:
            level = 1

        # Use warnings for level = 0
        if level == 0:

            # Conv print to str - then warn
            sep = ' '
            if 'sep' in kwargs:
                sep = kwargs.pop('sep')
            as_str = sep.join(str(arg) for arg in args)
            warnings.warn(as_str)

        if verbose >= level:
            print(*args, **kwargs, flush=True)

    return _print
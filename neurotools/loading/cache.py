import os
from ..misc.text import readable_size_to_bytes
from pathlib import Path

from joblib.hashing import hash as joblib_hash

def _get_cache_dr(cache_dr, load_func):

    # Process default
    if cache_dr == 'default':
        func_name = load_func.__name__
        default_cache_dr = os.path.join(os.path.expanduser("~"), 'neurotools_cache')
        cache_dr = os.path.join(default_cache_dr, func_name)

    # Give error if exists and not directory
    if os.path.exists(cache_dr) and not os.path.isdir(cache_dr):
        raise RuntimeError(f'Passed cache_dr: {cache_dr} already exists and is not a directory!')

    # Make sure exists if doesn't exist
    os.makedirs(cache_dr, exist_ok=True)

    return cache_dr

def _get_load_hash_str(load_args, use_base_name):

    to_cache_args = []
    for arg in load_args:

        # None case
        if arg is None:
            to_cache_args.append(None)

        # If str, check if location, if location use
        # basename instead of absolute path
        elif use_base_name and isinstance(arg, str) and os.path.exists(arg):
            to_cache_args.append(os.path.basename(arg))

        # If list, sort
        elif isinstance(arg, list):
            to_cache_args.append(sorted(arg))

        # Otherwise add directly
        else:
            print(arg)
            to_cache_args.append(arg)

    # Return joblib hash
    return joblib_hash(to_cache_args)

def _get_load_cache_loc(load_args, cache_dr, use_base_name):

    # Otherwise, caching behavior is expected
    # Get unique hash for these arguments - with tolerance to order and
    # and a few other things.
    hash_str = _get_load_hash_str(load_args, use_base_name)

    # Determine loc from hash_str
    cache_loc = os.path.join(cache_dr, hash_str)

    return cache_loc

def _unpack_cache_args(cache_args):

    if 'cache_dr' in cache_args:
        cache_dr = cache_args['cache_dr']
    else:
        cache_dr = None

    if 'cache_max_sz' in cache_args:
        cache_max_sz = cache_args['cache_max_sz']
    else:
        cache_max_sz = '30G'

    if 'use_base_name' in cache_args:
        use_base_name = cache_args['use_base_name']
    else:
        use_base_name = True

    # TODO warn if any others passed and not used

    return cache_dr, cache_max_sz, use_base_name

def _base_cache_load(load_args, load_func,
                     cache_load_func, cache_save_func,
                     cache_args, _print):

    # Unpack cache args
    cache_dr, cache_max_sz, use_base_name = _unpack_cache_args(cache_args)

    # If no caching, base case, just load as normal
    if cache_dr is None:
        _print(f'No cache_dr specified, loading directly', level=1)
        return load_func(*load_args)

    # Make sure cache_dr arg
    cache_dr = _get_cache_dr(cache_dr, load_func)

    # Get cache loc
    cache_loc = _get_load_cache_loc(load_args, cache_dr, use_base_name)

    # If exists, load from saved
    if os.path.exists(cache_loc):
        _print(f'Loading from cache_loc: {cache_loc}', level=1) 

        # Load with correct function
        data = cache_load_func(cache_loc, load_args)

        # Make sure to updating last modified of file to now
        Path(cache_loc).touch()

    # If doesn't yet exist, load like normal
    else:
        _print(f'No existing cache found, loading directly', level=1)

        # Base load
        data = load_func(*load_args)

        # Save with special cache func
        cache_save_func(data=data, cache_loc=cache_loc)

        _print(f'Saving loaded data to cache_loc: {cache_loc}', level=1)

    # Before returning - check the cache_max_sz argument
    # clearing any files over the limit
    _check_cache_sz_limit(cache_dr, cache_max_sz, _print=_print)

    return data

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
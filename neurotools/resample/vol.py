from neuromaps import transforms
from ..loading.from_data import get_surf_loc, _load_medial_wall
from ..loading.funcs import load
from ..transform.space import _resolve_space_name


def vol_to_surf(vol, target_space='32k_fs_LR',
                method='auto', ref_key='pial',
                **kwargs):
    '''

    The ref argument is only used with the nilearn style projection.

    If using the nilearn style vol to surf projection, the default
    values which can be overridden via kwargs are:

    radius=3.0, interpolation='linear', kind='auto',
    n_samples=None, mask_img=None,
    inner_mesh=None, depth=None.
    
    
    '''

    # Make sure valid target space
    target_space = _resolve_space_name(target_space)

    if method not in ['auto', 'nilearn', 'hcp']:
        raise RuntimeError('method must be one of auto, nilearn, hcp.')

    # Auto just tries hcp with tolerance for failure
    if method == 'auto':

        try:
            return _hcp_vol_to_surf(vol, target_space)
        except:
            print('Using nilearn resample.')

    # Only different is if error and hcp requested
    # pass along error.
    elif method == 'hcp':
        return _hcp_vol_to_surf(vol, target_space)
    
    # Base case is nilearn
    return _nilearn_vol_to_surf(vol, target_space, ref_key, **kwargs)
    

def _nilearn_vol_to_surf(vol, target_space, ref_key, **kwargs):

    from nilearn.surface import vol_to_surf

    projected = {}
    for hemi in ['lh', 'rh']:
        surf_mesh = get_surf_loc(space=target_space,
                                 hemi=hemi, key=ref_key)
        projected[hemi] = vol_to_surf(vol, surf_mesh=surf_mesh, **kwargs)

    return projected

def _get_fs_density(target_space):

    sz = len(_load_medial_wall(target_space))

    # Try to get density from base library
    try:
        return transforms.DENSITY_MAP[sz]
    except KeyError:
        raise RuntimeError(f'target_space: {target_space} not supported with hcp style resampling.')

def _hcp_vol_to_surf(vol, target_space):

    if 'fs_LR' in target_space:
        ret = transforms.mni152_to_fslr(vol, fslr_density=target_space.split('_')[0])
    elif 'fsaverage' in target_space:
        ret = transforms.mni152_to_fsaverage(vol, fsavg_density=_get_fs_density(target_space))
    elif 'civet' in target_space:
        ret = transforms.mni152_to_civet(vol, civet_density='41k')
    else:
        raise RuntimeError(f'target_space: {target_space} not supported with hcp style resampling.')

    # Convert return to loaded style
    return {'lh': load(ret[0]), 'rh': load(ret[1])}

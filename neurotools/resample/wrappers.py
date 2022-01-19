
import os
import nibabel as nib
from tempfile import gettempdir
import random
import numpy as np
from ..loading import load
import shutil
from nibabel.freesurfer.io import read_geometry
from ..transform.formats import geo_to_gifti, data_to_gifti
from .. import data_dr

resample_dr = os.path.join(data_dr, 'resample_fsaverage')

def _proc_temp_dr(temp_dr):

    if temp_dr is None:

        # Ensure random temp directory
        np.random.seed(random.randint(1, 10000000))
        temp_dr = os.path.join(gettempdir(), str(np.random.random()))
        os.makedirs(temp_dr)

    return temp_dr

def gen_cheap_midthickness(fs_subj, hemi, temp_dr=None):
    '''The “correct” way to generate a midthickness file is something like
    mris_expand -thickness lh.white 0.5 graymid.
    In a surf/ directory, this will create a lh.graymid file.
    The cheap way to generate a midthickness file
    is to average the coordinates from white and pial.

    This method does it the cheap way, then saves as gifti to the temp dr.
    
    '''

    temp_dr = _proc_temp_dr(temp_dr)

    # If str, assume directory if dict, paths
    if isinstance(fs_subj, str):
        fs_white = os.path.join(fs_subj, 'surf', f'{hemi}.white')
        fs_pial = os.path.join(fs_subj, 'surf', f'{hemi}.pial')
    elif isinstance(fs_subj, dict):
        fs_white, fs_pial = fs_subj['fs_white'], fs_subj['fs_pial']
    else:
        raise RuntimeError('Must pass fs_subj as directory loc or dict.')
    
    # Load
    white = read_geometry(fs_white)
    pial = read_geometry(fs_pial)
    
    # Get averaged coords
    mid_thickness_coords = np.average([white.coordinates, pial.coordinates], axis=0)
    
    # Conv to gifti
    mid_thickness_gii = geo_to_gifti(mid_thickness_coords, white.faces)
    
    # Save
    mid_thickness_out_loc = os.path.join(temp_dr, f'{hemi}.midthickness.surf.gii')
    nib.save(mid_thickness_gii, mid_thickness_out_loc)
    
    # Return location
    return mid_thickness_out_loc

def _geo_to_gii(in_loc, out_loc):

    in_geo = read_geometry(in_loc)
    in_geo_gii = geo_to_gifti(in_geo[0], in_geo[1])
    nib.save(in_geo_gii, out_loc)

def _get_curr_new_sphere_locs(fs_subj, hemi, target_res, temp_dr):
    '''Get as gifti - saving to temp dr if needed'''

    # If str, assume directory if dict, paths
    if isinstance(fs_subj, str):
        current_fs_sphere = os.path.join(fs_subj, 'surf', f'{hemi}.sphere.reg')
    elif isinstance(fs_subj, dict):
        current_fs_sphere = fs_subj['current_fs_sphere']
    else:
        raise RuntimeError('Must pass fs_subj as directory loc or dict.')

    # Convert current fs sphere to gifti and save
    current_gii_sphere_loc = os.path.join(temp_dr, f'{hemi}.sphere.reg.surf.gii')
    _geo_to_gii(current_fs_sphere, current_gii_sphere_loc)
    
    # Get location of the sphere to sample to
    new_sphere_loc = os.path.join(resample_dr, f'fs_LR-deformed_to-fsaverage.{hemi[0].upper()}.sphere.{target_res}_fs_LR.surf.gii')

    return current_gii_sphere_loc, new_sphere_loc

def freesurfer_resample_prep(fs_subj, hemi,
                             target_res='32k',
                             midthickness_loc=None,
                             temp_dr=None):
    '''This method for now requires that the hcp workbench be installed.'''
    
    temp_dr = _proc_temp_dr(temp_dr)
    
    # If not passed, generate with cheap method
    if midthickness_loc is None:
        midthickness_loc = gen_cheap_midthickness(fs_subj=fs_subj,
                                                  hemi=hemi,
                                                  temp_dr=temp_dr)
    
    # TODO process if midthickness is not already gifti
    
    # Get cur + new_sphere locs
    current_gii_sphere_loc, new_sphere_loc =\
        _get_curr_new_sphere_locs(fs_subj=fs_subj, hemi=hemi, target_res=target_res, temp_dr=temp_dr)

    # Get location of where to save sampled mid thickness
    midthickness_new_out_loc = os.path.join(temp_dr, f'{hemi}.midthickness.{target_res}_fs_LR.surf.gii')

    # Run command
    os.system(f'wb_command -surface-resample {midthickness_loc} {current_gii_sphere_loc} {new_sphere_loc} BARYCENTRIC {midthickness_new_out_loc}')

    # Return dict of created files
    return {'current_gii_sphere_loc': current_gii_sphere_loc,
            'midthickness_new_out_loc': midthickness_new_out_loc,
            'midthickness_loc': midthickness_loc,
            'new_sphere_loc': new_sphere_loc}

def _save_data_as_gii(data, temp_dr=None, append=''):

    temp_dr = _proc_temp_dr(temp_dr)
    
    # If data is list special case
    if isinstance(data, list):
        data_locs = [_save_data_as_gii(d, temp_dr, append=i) for i, d in enumerate(data)]
        return data_locs
    
    # Make sure loaded
    data = load(data)

    # Convert from array to gifti
    data_gii = data_to_gifti(data)

    # Save in temp dr
    data_loc = os.path.join(temp_dr, f'data{append}.gii')
    nib.save(data_gii, data_loc)

    return data_loc

def resample_fs_native_to_fs_LR(data, fs_subj, hemi,
                                target_res='32k',
                                midthickness_loc=None,
                                area_correction=True,
                                temp_dr=None):
    
    # Init temp dr
    temp_dr = _proc_temp_dr(temp_dr)
    
    # Load data to sample as array
    # then save as gifti
    # If data is explicitly a list, data loc will return a list
    data_loc = _save_data_as_gii(data, temp_dr)

    # Where to save result of re-sample
    metric_out_loc = os.path.join(temp_dr, 'metric_out.func.gii')

    if area_correction:

        # Call freesurfer prep to generate files for resample
        files = freesurfer_resample_prep(fs_subj=fs_subj, hemi=hemi,
                                         target_res=target_res,
                                         midthickness_loc=midthickness_loc,
                                         temp_dr=temp_dr)
    
        # Generate command
        cmd = f'wb_command -metric-resample DATA_LOC {files["current_gii_sphere_loc"]} {files["new_sphere_loc"]} ADAP_BARY_AREA {metric_out_loc} -area-surfs {files["midthickness_loc"]} {files["midthickness_new_out_loc"]}'

    else:

        # Get cur + new_sphere locs
        current_gii_sphere_loc, new_sphere_loc =\
        _get_curr_new_sphere_locs(fs_subj=fs_subj, hemi=hemi, target_res=target_res, temp_dr=temp_dr)

        # Generate command
        cmd = f'wb_command -metric-resample DATA_LOC {current_gii_sphere_loc} {new_sphere_loc} BARYCENTRIC {metric_out_loc}'
    
    # If processing multiple
    if isinstance(data_loc, list):

        resampled_data = []
        for d_loc in data_loc:
        
            # Clear existing in case of errors
            if os.path.exists(metric_out_loc):
                os.remove(metric_out_loc)
            
            # Run re-sampling
            os.system(cmd.replace('DATA_LOC', d_loc))
            
            # Load to resampled data
            resampled_data.append(load(metric_out_loc))
    
    # Single data point
    else:

        # Run re-sampling
        os.system(cmd.replace('DATA_LOC', data_loc))
        
        # Load resamples
        resampled_data = load(metric_out_loc)

    # Clean up temp directory
    shutil.rmtree(temp_dr)
    
    # Return the re-sampled data
    return resampled_data
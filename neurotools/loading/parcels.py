import os
import numpy as np
from .. import data_dr

def load_32k_fs_LR_concat(parcel_name):
    '''Dedicated loader function for saved parcels as generated in
    parc_scaling project. These parcellations are all in space 32k_fs_LR,
    and are left, right hemisphere concatenated. The parcel will be
    downloaded to parcels directory in the default data dr.

    Parameters
    -----------
    parcel_name : str
        The name of the parcel to load,
        see https://github.com/sahahn/parc_scaling/tree/main/parcels
        for valid options.
    
    Returns
    -------
    parcel : numpy array
        The loaded concat LR parcellation as a numpy array is returned.
    '''

    # Tolerance to passing with or without npy tag
    parcel_name = parcel_name.replace('.npy', '')
    
    # Make sure parcel directory init'ed
    parcel_dr = os.path.join(data_dr, 'parcels', '32k_fs_LR_concat')
    os.makedirs(parcel_dr, exist_ok=True)
    
    # Get location
    parcel_loc = os.path.join(parcel_dr, f'{parcel_name}.npy')
    
    # If doesn't exist, try to download
    if not os.path.exists(parcel_loc):
        print('Downloading parcel...')
        parcel_url = f'https://raw.githubusercontent.com/sahahn/parc_scaling/main/parcels/{parcel_name}.npy'
        os.system(f'wget -L {parcel_url} -P {parcel_dr}')

    # Load parcel
    parcel = np.load(parcel_loc)

    return parcel
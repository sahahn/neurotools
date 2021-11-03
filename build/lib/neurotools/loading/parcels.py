import os
import numpy as np
from .. import data_dr

def load_32k_fs_LR_concat(parcel_name):
    '''Dedicated loaded function for saved parcels as generated in
    parc_scaling project.'''
    
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
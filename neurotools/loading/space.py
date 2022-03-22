import os

from .. import data_dr

def get_space_options():
    
    spaces = []
    
    for file in os.listdir(data_dr):
        if file.startswith('_'):
            continue
        if file in ['resample_fsaverage', 'mappings', 'parcels', 'index_maps']:
            continue
        
        spaces.append(file)
    
    return spaces
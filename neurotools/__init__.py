from pathlib import Path
import os
from urllib import request
import tarfile

# Set version
__version__ = '0.24'
__data_version__ = '1.2.7'

CURRENT_DR = Path(__file__).parent.absolute()
DATA_REF_LOC = os.path.join(CURRENT_DR, 'data_ref.txt')

def get_data_dr_path(download_loc):
    return os.path.join(download_loc, f'neurotools_data-{__data_version__}', 'data')

def download(download_loc):

    print(f'Downloading latest neurotools_data to {download_loc}')
    zipped_file = f'{download_loc}.tar.gz'

    # Download
    request.urlretrieve(f'https://github.com/sahahn/neurotools_data/archive/refs/tags/{__data_version__}.tar.gz', zipped_file)

    # Make sure main directory exists
    os.makedirs(download_loc, exist_ok=True)

    # Un-zip to main directory
    with tarfile.open(zipped_file) as f:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, download_loc)

    # Remove tar gz file once done
    os.remove(zipped_file)

    # Save location of the main neurotools data file 
    with open(DATA_REF_LOC, 'w') as f:
        f.write(download_loc)

    # This is the location of the download data
    data_dr = get_data_dr_path(download_loc)

    print(f'Downloaded data version = {__data_version__} complete!')
    print(f'Current version saved at: {data_dr}')
    print(f'If you move this directory, make sure to update saved location in data ref at {DATA_REF_LOC}.')

    return data_dr

def check_DATA_REF_LOC():

    # If doesn't exist, init
    if not os.path.exists(DATA_REF_LOC):
        
        # Set to default download loc if doesn't already exist
        # TODO could prompt user instead?
        home_dr = str(Path.home())
        download_loc = os.path.join(home_dr, 'neurotools_data')

        # Save default
        with open(DATA_REF_LOC, 'w') as f:
            f.write(download_loc)

def resolve_data_dr():

    # Make sure data ref has been init'ed
    check_DATA_REF_LOC()

    # Check saved    
    with open(DATA_REF_LOC, 'r') as f:
        download_loc = f.readline().rstrip()

    # If exists, check to see if latest version exists
    data_dr = get_data_dr_path(download_loc)
    if os.path.exists(data_dr):
        return data_dr

    # Otherwise, download latest
    return download(download_loc)

# Resolve data dr, downloading if needed
data_dr = resolve_data_dr()



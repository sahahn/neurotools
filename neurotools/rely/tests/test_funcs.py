import numpy as np
import os
import pandas as pd
from ...loading.funcs import load
from ..funcs import run_rely

file_dr = os.path.dirname(os.path.realpath(__file__))

def setup_fake_data():

    # Save fake data
    fake_dr = os.path.join(file_dr, 'fake_data')
    os.makedirs(fake_dr, exist_ok=True)

    for i in range(100):
        fake_data = np.random.random(10)
        np.save(os.path.join(fake_dr, str(i) + '.npy'), fake_data)
    
    # Gen covars df
    covars_df = pd.DataFrame(np.random.random((100, 2)))

    return fake_dr, covars_df


def test_rely_basic():
    
    # Setup fake data
    fake_dr, covars_df = setup_fake_data()

    # Just shouldn't fail
    run_rely(covars_df,
             template_path=os.path.join(fake_dr, 'SUBJECT.npy'),
             max_size=30, verbose=-1)
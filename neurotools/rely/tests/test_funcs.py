import numpy as np
import os
import pandas as pd
from ...loading.funcs import load
from ..funcs import run_rely, _test_split

file_dr = os.path.dirname(os.path.realpath(__file__))

def setup_fake_data():

    fake_dr = os.path.join(file_dr, 'fake_data')
    
    # Only save if doesnt exist
    if not os.path.exists(fake_dr):
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


def test_rely_basic_data_df():

    covars_df = pd.DataFrame(np.random.random((100, 2)))
    data_df = pd.DataFrame(np.random.random((100, 20)))

    run_rely(covars_df,
             data_df=data_df,
             max_size=30, verbose=-1)

def test_test_split_groups():

    all_subjects = ['s1', 's2', 's3', 's4', 's5']
    stratify = None
    groups = pd.Series([1, 1, 2, 2, 3], index=all_subjects)

    for split in range(50):

        g1, g2 = _test_split(all_subjects, stratify,
                             groups, split_random_state=split)

        if 's1' in g1:
            assert 's2' in g1
        elif 's1' in g2:
            assert 's2' in g2

        if 's3' in g1:
            assert 's4' in g1
        elif 's3' in g2:
            assert 's4' in g2


def test_test_split_stratify():

    all_subjects = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
    stratify = pd.Series([1, 1, 1, 1, 0, 0, 0, 0], index=all_subjects)
    groups = None

    for split in range(50):

        g1, g2 = _test_split(all_subjects, stratify,
                             groups, split_random_state=split)
        assert np.sum(stratify.loc[g1]) == np.sum(stratify.loc[g2])

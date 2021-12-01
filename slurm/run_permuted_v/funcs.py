from sklearn.utils import check_random_state
import os
import random
import time
import numpy as np
import pickle as pkl
from neurotools.stats.permutations import _process_permuted_v_base

def setup_and_run_permuted_v(results_dr,
                             tested_vars,
                             target_vars,
                             confounding_vars,
                             permutation_structure,
                             n_perm,
                             two_sided_test=True,
                             within_grp=True,
                             random_state=None,
                             use_tf=False,
                             n_jobs=1,
                             dtype=None,
                             use_z=False,
                             demean_confounds=True,
                             use_short_jobs=True,
                             job_mem='4G',
                             job_factor=3,
                             sz_target=10000):

    # Make sure dr's init'ed - just error out if results dr already exists
    os.makedirs(results_dr, exist_ok=False)
    jobs_logs_dr = os.path.join(results_dr, 'Job_Logs')
    os.makedirs(jobs_logs_dr, exist_ok=True)

    # Init temp directory - should not already exist
    temp_dr = os.path.join(results_dr, 'temp_' + str(random.random()))
    os.makedirs(temp_dr)

    # Decide on partition
    if use_short_jobs:
        limit = 10800
        j_time = '3:00:00'
        partition = 'short'
    
    # Non-short case
    else:
        limit = 108000
        j_time = '30:00:00'
        partition = 'bluemoon'

    # Idea is here could optionally split target vars by data dimension
    n_splits = max([target_vars.shape[-1] // sz_target, 1])
    target_vars_parts = np.array_split(target_vars, n_splits, axis=-1)

    # If job_mem is passed without size indicator
    try:

        # Will trigger except if passed with size indicator here
        job_mem = int(job_mem)

        # Set to heuristic lower memory than original submission
        job_mem = str(int(job_mem / max([n_splits / 2, 1])))
    except:
        pass

    # For each target part, process and save base
    # TODO multiple process this piece? i.e., submit jobs?
    times, original_scores, original_scores_sign = [], [], []
    for i, target_vars_part in enumerate(target_vars_parts):
        print(f'Start proc split {i} with shape {target_vars_part.shape}', flush=True)

        # Idea is to run the initial processing + first base permutation,
        # which help us get an idea at runtime also.
        start = time.time()
        base =\
            _process_permuted_v_base(
                tested_vars=tested_vars, target_vars=target_vars_part,
                confounding_vars=confounding_vars,
                permutation_structure=permutation_structure,
                n_perm=n_perm, two_sided_test=two_sided_test,
                within_grp=within_grp, random_state=random_state,
                use_tf=use_tf, n_jobs=n_jobs, dtype=dtype,
                use_z=use_z, demean_confounds=demean_confounds)

        # Put only the relevant args in dict, then save with pickle
        args = {'run_perm_func': base[2], 'target_vars': base[3],
                'rz': base[4], 'hz': base[5], 'input_matrix': base[6],
                'variance_groups': base[7], 'drm': base[8], 'contrast': base[9],
                'rng': base[10], 'permutation_structure': permutation_structure,
                'use_z': use_z, 'two_sided_test': two_sided_test,
                'n_splits': n_splits, 'limit': limit, 'n_perm': n_perm}

        with open(os.path.join(temp_dr, f'args_{i}.pkl'), 'wb') as f:
            pkl.dump(args, f)

        # Append time
        times.append(time.time() - start)

        # Keep track of original scores and sign
        original_scores.append(base[0])
        original_scores_sign.append(base[1])

    # Calculate upper time limit from mean time
    time_est = np.mean(times) * job_factor
    realistic_time_est = np.max(times)
    print('Estimate mean time * job_factor used for submitting jobs is:', time_est, flush=True)
    print('Realistic max time est:', np.max(times), flush=True)

    # Calculate how many permutations each
    # submitted job could safely run, to estimate
    # how many jobs to submit
    per_job_n_perm = limit / time_est
    est_jobs_needed = int(np.ceil(n_perm / per_job_n_perm))

    # The output and error file locs
    o, e = os.path.join(jobs_logs_dr, '%x_%j.out'), os.path.join(jobs_logs_dr, '%x_%j.err')

    # The jobs w/ extra 1/5th as filler
    n_filler_jobs = n_splits // 5
    print(f'Submitting {n_filler_jobs} filler jobs', flush=True)
    arg_n_splits = list(range(n_splits)) + [-1 for _ in range(n_filler_jobs)]

    # For each data split part - submit jobs
    for arg_n in arg_n_splits:

        s1_cmd = f'sbatch --array=1-{est_jobs_needed} --mem={job_mem} --output={o} --error={e} '
        s2_cmd = f'--partition={partition} --time={j_time} run_permuted_v.sh '
        s3_cmd = f'{temp_dr} {arg_n} {realistic_time_est}'
        submit_cmd = s1_cmd + s2_cmd + s3_cmd
        
        print(submit_cmd, flush=True)
        os.system(submit_cmd)

    # Save a version of the full original scores
    original_scores = np.concatenate(original_scores)
    if two_sided_test:
        original_scores *= np.concatenate(original_scores_sign)

    save_loc = os.path.join(results_dr, 'original_scores.npy')
    np.save(save_loc, original_scores)
    



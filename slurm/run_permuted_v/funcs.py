from neurotools.loading.abcd import load_from_csv, load_family_block_structure
from neurotools.loading.funcs import get_overlap_subjects, load_data
from neurotools.loading.outliers import drop_top_x_outliers
import os
import random
import time
import numpy as np
import pickle as pkl
from neurotools.stats.permutations import _process_permuted_v_base

def get_results_dr(results_dr):

    if not os.path.exists(results_dr):
        return results_dr

    # Find next free
    i = 0
    while os.path.exists(new_results_dr := f'{results_dr}_{i}'):
        i += 1

    return new_results_dr

def init_drs(results_dr):

    # Get results_dr, if already exists, append
    results_dr = get_results_dr(results_dr)

    # Make sure dr's init'ed - just error out if results dr already exists
    os.makedirs(results_dr, exist_ok=False)

    jobs_logs_dr = os.path.join(results_dr, 'Job_Logs')
    os.makedirs(jobs_logs_dr, exist_ok=True)

    # Init temp directory - should not already exist
    # Temp directory is used to make cleaning up easier
    temp_dr = os.path.join(results_dr, 'temp_' + str(random.random()))
    os.makedirs(temp_dr)

    return results_dr, jobs_logs_dr, temp_dr

def proc_job_partition(use_short_jobs):

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

    return limit, j_time, partition

def proc_time_ests(times, job_factor, limit, n_perm, per_split_limit):

    # Calculate upper time limit from mean time
    time_est = np.mean(times) * job_factor
    realistic_time_est = np.max(times)
    print('Estimate mean time * job_factor used for submitting jobs is:', time_est, flush=True)
    print('Realistic max time est. per data split:', np.max(times), flush=True)

    # Calculate how many permutations each
    # submitted job could safely run, to estimate
    # how many jobs to submit
    per_job_n_perm = limit / time_est
    est_jobs_needed = int(np.ceil(n_perm / per_job_n_perm))

    # If estimate over per split limit, set to per split limit
    est_jobs_needed = min([est_jobs_needed, per_split_limit])

    return est_jobs_needed, realistic_time_est

def proc_job_mem(job_mem, temp_dr):

    # Get job mem from arguments size
    if job_mem is None:

        # Size of base arguments in bytes
        base_sz_bytes = os.path.getsize(os.path.join(temp_dr, 'base_args.pkl'))

        # Size of first split in bytes
        target_split_sz_bytes = os.path.getsize(os.path.join(temp_dr, 'target_vars_0.npy'))

        # Multiply by scaler, then divide so that is kb
        job_mem =  int((base_sz_bytes + target_split_sz_bytes) * 4 / 1024)

        # Put as str with size indicator
        job_mem = f'{job_mem}K'

    return job_mem

def save_original_scores(original_scores, original_scores_sign,
                         two_sided_test, results_dr):

    # Get original scores w/ correct sign
    original_scores = np.concatenate(original_scores)
    if two_sided_test:
        original_scores *= np.concatenate(original_scores_sign)

    # Save a version of the full original scores
    save_loc = os.path.join(results_dr, 'original_scores.npy')
    np.save(save_loc, original_scores)

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
                             dtype='float32',
                             use_z=False,
                             demean_confounds=True,
                             use_short_jobs=True,
                             job_mem=None,
                             job_factor=3,
                             sz_target=10000,
                             job_limit=500,
                             min_vg_size=None):

    # Process initial directories
    results_dr, jobs_logs_dr, temp_dr = init_drs(results_dr)

    # Set based on if short or not
    limit, j_time, partition = proc_job_partition(use_short_jobs)

    # Idea is here could optionally split target vars by data dimension
    n_splits = max([target_vars.shape[-1] // sz_target, 1])
    target_vars_parts = np.array_split(target_vars, n_splits, axis=-1)
    print(f'Base: {n_splits} data splits', flush=True)

    # The jobs w/ extra 1/5th as filler
    n_filler_jobs = n_splits // 5
    print(f'Extra: {n_filler_jobs} filler jobs', flush=True)
    arg_n_splits = list(range(n_splits)) + [-1 for _ in range(n_filler_jobs)]

    # Calculate new per job filler limit based on n_splits + filler jobs
    per_split_limit = job_limit // (n_splits + n_filler_jobs)

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
                use_z=use_z, demean_confounds=demean_confounds,
                min_vg_size=min_vg_size)

        # Most of the arguments, only need to save once / first time
        if i == 0:
            base_args = {'run_perm_func': base[2], 'rz': base[4],
                         'hz': base[5], 'input_matrix': base[6],
                         'permutation_structure': base[7],
                         'variance_groups': base[8], 'drm': base[9],
                         'contrast': base[10], 'rng': base[11],
                         'intercept_test': base[12],
                         'use_z': use_z, 'two_sided_test': two_sided_test,
                         'n_splits': n_splits, 'limit': limit, 'n_perm': n_perm}

            with open(os.path.join(temp_dr, f'base_args.pkl'), 'wb') as f:
                pkl.dump(base_args, f)

        # Need to save target vars for each split though - use numpy
        np.save(os.path.join(temp_dr, f'target_vars_{i}.npy'), base[3])

        # Append time
        times.append(time.time() - start)

        # Keep track of original scores and sign
        original_scores.append(base[0])
        original_scores_sign.append(base[1])

    print(flush=True)

    # Get estimated number of jobs and time est
    est_jobs_needed, realistic_time_est =\
        proc_time_ests(times, job_factor, limit, n_perm, per_split_limit)

    # The output and error file locs
    o, e = os.path.join(jobs_logs_dr, '%x_%j.out'), os.path.join(jobs_logs_dr, '%x_%j.err')

    print(flush=True)

    # Get job mem estimate if not explicitly passed
    job_mem = proc_job_mem(job_mem, temp_dr)

    # For each data split part - submit jobs
    for arg_n in arg_n_splits:

        s1_cmd = f'sbatch --array=1-{est_jobs_needed} --mem={job_mem} --output={o} --error={e} '
        s2_cmd = f'--partition={partition} --time={j_time} run_permuted_v.sh '
        s3_cmd = f'{temp_dr} {arg_n} {realistic_time_est}'
        submit_cmd = s1_cmd + s2_cmd + s3_cmd
        
        print(submit_cmd, flush=True)
        os.system(submit_cmd)

    # Save original scores w/ correct sign
    save_original_scores(original_scores, original_scores_sign,
                         two_sided_test, results_dr)


def load_and_setup_data(tested_vars, confounding_vars, csv_loc,
                        eventname, template_path, mask,
                        dtype, drop_top):

    # Load target + confounds, dummy coded + no nans
    df = load_from_csv(cols=tested_vars + confounding_vars,
                       csv_loc=csv_loc,
                       eventname=eventname,
                       drop_nan=True,
                       encode_cat_as='dummy',
                       verbose=1)
    print('Loaded initial df', df.shape, flush=True)
    print(flush=True)

    # Update confounding vars to reflect names after dummy coding
    confounding_vars = list(df)
    confounding_vars.remove(tested_vars[0])

    # Get overlap of subjects with df and imaging data
    subjects = get_overlap_subjects(df=df,
                                    template_path=template_path,
                                    verbose=1)

    # Load default permutation structure - we do this
    # twice, but first here to find if any missing values
    permutation_structure = load_family_block_structure(
                                csv_loc=csv_loc, 
                                subjects=subjects,
                                eventname=eventname,
                                verbose=0)
    print(flush=True)


    # Get latest overlap of subjects as permutation structure df index
    subjects = permutation_structure.index

    # Load in the imaging data
    target_vars = load_data(subjects=subjects,
                            template_path=template_path,
                            mask=mask,
                            nan_as_zero=True,
                            dtype=dtype,
                            n_jobs=4, verbose=1)

    # Perform optional outlier filtering
    target_vars, subjects = drop_top_x_outliers(target_vars, subjects, drop_top)
    print(f'Loaded Max-Min after drop: {drop_top} outliers {np.max(target_vars)} {np.min(target_vars)}', flush=True)
    print(flush=True)

    # Re-load permutation block structure as subjects have changed
    permutation_structure = load_family_block_structure(
                                csv_loc=csv_loc, 
                                subjects=subjects,
                                eventname=eventname,
                                verbose=1)

    # Should load the same number of subjects here
    assert len(permutation_structure) == len(subjects)

    # Make sure df matches
    df = df.loc[subjects]

    # Return as numpy arrays
    return (np.array(df[tested_vars]), target_vars,
            np.array(df[confounding_vars]), np.array(permutation_structure))
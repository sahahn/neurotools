import os
import time
import pandas as pd
import pickle as pkl
import sys
from config import get_job_limit
import numpy as np
from statsmodels.stats.multitest import multipletests
from neurotools.stats import run_mixed_model
import random
import shutil


def run_slurm_mm(df, data, fixed_effects_vars,
                 random_effects_vars, results_loc,
                 use_short_jobs=True, job_mem='1G'):
    
    # Make sure dr's init'ed
    os.makedirs('Job_Logs', exist_ok=True)
    os.makedirs(results_loc, exist_ok=True)

    # Init temp directory should not exist
    temp_dr = os.path.join(results_loc, 'temp_' + str(random.random()))
    os.makedirs(temp_dr)

    # Save df
    df.to_pickle(os.path.join(temp_dr, 'df.pkl'))

    # Save data by each vertex / vox for
    # lower memory req. per job
    data_dr = os.path.join(temp_dr, 'data')
    os.makedirs(data_dr, exist_ok=True)
    for i in range(data.shape[1]):
        save_loc = os.path.join(data_dr, str(i) + '.npy')
        np.save(save_loc, data[:, i])

    # Save args
    args = {'random_effects_vars': random_effects_vars,
            'fixed_effects_vars': fixed_effects_vars,
            'data_shape1': data.shape[1],
            'results_loc': results_loc}

    with open(os.path.join(temp_dr, 'args.pkl'), 'wb') as f:
        pkl.dump(args, f)

    # Run once to get time estimate
    time_est = single_run(temp_dr, df, args)
    print(time_est, flush=True)

    # Use 2.5 the example run for the estimate
    # Can adjust this if doesn't seem to be actually
    # finishing - over submitting isn't really a problem either.
    # Just better to avoid over using resources.
    time_est = time_est * 2.5
    limit = get_job_limit(short=use_short_jobs)

    each_job = limit // time_est
    jobs_needed = (data.shape[1] // each_job) + 1

    if use_short_jobs:
        script_name = 'run_short_mm.sh'
    else:
        script_name = 'run_mm.sh'

    # Submit array of jobs to finish
    print()
    submit_cmd = f'sbatch --array=1-{int(jobs_needed)} --mem={job_mem} {script_name} {temp_dr}'
    os.system(submit_cmd)

    print(f'Submitted {int(jobs_needed)} jobs.')
    print('If this is not enough and more jobs need to be run, you can submit')
    print('more jobs by modifying the following command:')
    print(submit_cmd)

    print()
    print(f'Final results when finished will be saved in directory: {results_loc}')

    print()
    print('In the case where jobs start failing with memory errors,')
    print('you must manually change the line in files run_mm.sh and/or')
    print('run_short_mm.sh "#SBATCH --mem=1G" to a higher value.')

def run_from_save(temp_dr, short=False):

    # Check to see if already fully done first
    check_already_done(temp_dr)
    
    # Setup time info
    start_time = time.time()

    # Get limit
    limit = get_job_limit(short=short)
    
    # Load in df - data is loaded later
    df = pd.read_pickle(os.path.join(temp_dr, 'df.pkl'))
    with open(os.path.join(temp_dr, 'args.pkl'), 'rb') as f:
        args = pkl.load(f)
    
    # Continue to loop while time remains
    while time.time() - start_time < limit:
        
        # Run once
        single_run(temp_dr, df, args)
        

def single_run(temp_dr, df, args):
    
    # Get from args
    data_shape1 = args['data_shape1']
    results_loc = args['results_loc']

    # Make sure results directory init'ed
    results_dr = os.path.join(temp_dr, 'results')
    os.makedirs(results_dr, exist_ok=True)

    # All possible combinations
    all_options = set(range(data_shape1))

    # Start by checking at every loop which still
    # need to be run
    finished = set(int(r) for r in os.listdir(results_dr))
    avaliable = all_options - finished

    # Check if finished
    if len(avaliable) == 0:
        print('All options finished!', flush=True)
        check_end_condition(temp_dr, results_loc)
        sys.exit()

    print(f'Remaining: {len(avaliable)}', flush=True)
    
    # Select option to run
    i = random.choice(tuple(avaliable))
    save_loc = os.path.join(results_dr, str(i))

    # Init results file as empty to indicate
    # to other jobs not to run this one
    with open(save_loc, 'w') as f:
        pass

    # Load in the correct data slice
    data = np.load(os.path.join(temp_dr, 'data', str(i) + '.npy'))

    # Run this vertex / vox
    job_start_time = time.time()
    df['data'] = data
    
    # Run mixed model - pass copy of df, as df can change
    result = run_mixed_model(df.copy(),
                             fixed_effects_vars=args['fixed_effects_vars'],
                             random_effects_vars=args['random_effects_vars'])

    # Why doesn't result.remove_data() work?
    save_results =\
         {'params': result.params,
          'pvalues': result.pvalues}
        
    # Save subset of results as pickle
    with open(save_loc, 'wb') as f:
        pkl.dump(save_results, f)

    print(f'Finished {i} in {time.time() - job_start_time}', flush=True)
    
    # Return time it took to run once
    return time.time() - job_start_time


def check_already_done(temp_dr):

    # If temp_dr already deleted, then everything already finished
    if not os.path.exists(temp_dr):
        print('Everything already finished.', flush=True)
        sys.exit()


def check_end_condition(temp_dr, results_loc):

    # Check to make sure not already fully done
    check_already_done(temp_dr)

    print('Checking end conditions.')

    # Get list of all completed files
    results_dr = os.path.join(temp_dr, 'results')
    result_files = [os.path.join(results_dr, f) for f in os.listdir(results_dr)]

    # If any not done - ignore
    result_file_sizes = [os.path.getsize(f) for f in result_files]
    if np.any(np.array(result_file_sizes) == 0):
        print('Some runs not finished, ending.')
        return

    # Get variables / column names
    with open(result_files[0], 'rb') as f:
        ex_r = pkl.load(f)
        fields = list(ex_r.keys())
        cols = list(ex_r['params'].index)

    # Init. condensed results
    results = {field: {col: np.zeros(shape=len(result_files))
                       for col in cols}
               for field in fields}
    
    # Fill in results
    for file in result_files:

        # Load
        with open(file, 'rb') as f:
            result = pkl.load(f)
        
        # Determine index from file name
        indx = int(file.split('/')[-1])
         
        # Fill in correct spot in results
        for field in fields:
            for col in cols:
                results[field][col][indx] = result[field].loc[col]

    # Add multiple comparisons corrected values
    results['corrected_pvalues'] = {col: multipletests(results['pvalues'][col],
                                                       alpha=0.05,
                                                       method='fdr_bh')[1] for col in cols}
    
    # Save results as pkl
    with open(os.path.join(results_loc, 'results.pkl'), 'wb') as f:
        pkl.dump(results, f)

    # Also save results in alternate csv-eqsue formats

    # Remove all temp results
    shutil.rmtree(temp_dr)


import sys
import pickle as pkl
import os
import numpy as np
import shutil
import time
import glob
from pathlib import Path
import random

from neurotools.stats.permutations import _get_perm_matrix

# Some tunable params here
EXTRA_PAD = 600
VERBOSE_EVERY = 25
CHECK_END_EVERY = 10
CHANGE_FILLER_EVERY = 20

# Keep track of number start time for verbose / filler jobs
# as global variable
script_start_time = time.time()

def unpack_args():

    # Unpack args
    args = list(sys.argv[1:])

    # Temp and results dr
    temp_dr = args[0]

    # The data dimension split
    arg_n = int(args[1])

    # Estimate per job from submitting job
    time_est = float(args[2])

    # Job array id
    job_id = int(args[3])

    return temp_dr, arg_n, time_est, job_id

def end_job_print(n_run):

    print('Job Finished.', flush=True)
    print(f'Completed {n_run} permutations in {(time.time() - script_start_time):.3f}')

def check_already_finished(results_dr, temp_dr, n_run):
    
    # If h0_vmax is saved, then we know already done
    if os.path.exists(os.path.join(results_dr, 'h0_vmax.npy')):

        # If pvals done and temp_dr still exists
        # try to help clean up temp dr before exit
        if os.path.exists(os.path.join(results_dr, 'pvals.npy')):
            if os.path.exists(temp_dr):
                shutil.rmtree(temp_dr, ignore_errors=True)

        # Print before finish / quit
        end_job_print(n_run)

        # Exit
        sys.exit()

def check_already_exists(temp_dr, arg_n, random_state):

    # First check final save loc
    if os.path.exists(get_merged_save_loc(temp_dr, random_state)):
        return True

    # Next check the intermediate save_loc
    if os.path.exists(get_split_save_loc(temp_dr, random_state, arg_n)):
        return True

    return False

def get_merged_save_loc(temp_dr, random_state, init_dr=False):
    '''This is the location of the saved h0 vmax across all data'''

    # If getting loc for save, we want to make dr is needed
    save_dr = os.path.join(temp_dr, 'results')
    if init_dr:
        os.makedirs(save_dr, exist_ok=True)

    # Save just under random state
    return os.path.join(save_dr, f'{random_state}.npy')

def get_split_save_loc(temp_dr, random_state, arg_n, init_dr=False):

    # Make sure the directory exists first if passed
    save_dr = os.path.join(temp_dr, 'temp_results', str(random_state))
    if init_dr:
        os.makedirs(save_dr, exist_ok=True)

    # Return save loc of this chunk of data's h0 vmax
    return os.path.join(save_dr, f'{arg_n}.npy')

def proc_perm_vmax(temp_dr, random_state, arg_n, h0_vmax, n_splits):

    # If n_splits is 1, means no data splits
    # were done, and we can save this result directly to merged_save_loc
    if n_splits == 1:
        
        merged_save_loc =\
            get_merged_save_loc(temp_dr, random_state, init_dr=True)
        np.save(merged_save_loc, h0_vmax)

        return

    # Otherwise, if there were splits then
    # check to see if saving this split
    # would finish across all data pieces
    split_save_locs = glob.glob(
        get_split_save_loc(temp_dr, random_state, arg_n='*'))

    # The current split save loc
    split_save_loc =\
        get_split_save_loc(temp_dr, random_state, arg_n, init_dr=True)

    # In this case, compute vmax across pieces
    if len(split_save_locs) == n_splits - 1:

        # There exists case where there may be n_splits - 1
        # but if the current split_save_loc already exists then
        # just return, as this is a duplicate case
        if os.path.exists(split_save_loc):
            print('SPLIT ALREADY EXISTED', random_state, arg_n, flush=True)
            return

        # Load all and get max of max's
        try:
            all_v_maxs = [np.load(loc) for loc in split_save_locs] + [h0_vmax]

        # Errors shouldn't happen often, but if they do
        # they are usually trying to read while writing
        except:

            # Essentially try one more time, then just skip
            try:
                all_v_maxs = [np.load(loc) for loc in split_save_locs] + [h0_vmax]

            # And if it still fails, then just skip this one
            except:
                return

        # Set to max across all
        merged_h0_vmax = np.max(all_v_maxs)

        # Merged save loc
        merged_save_loc =\
            get_merged_save_loc(temp_dr, random_state, init_dr=True)
        np.save(merged_save_loc, merged_h0_vmax)

        # Can remove the other intermediate files
        # not strictly necessary, but reduces burden
        # for last job in having to clean up everything
        old_dr = os.path.join(temp_dr, 'temp_results', str(random_state))
        shutil.rmtree(old_dr, ignore_errors=True)

        return
    
    # Otherwise, just save split vmax, if already
    # exists, just skip
    if os.path.exists(split_save_loc):
        print('SPLIT ALREADY EXISTED', random_state, arg_n, flush=True)
        return

    np.save(split_save_loc, h0_vmax)

    return

def check_end_condition(results_dr, temp_dr, n_perm, two_sided_test):

    # With globbing, get the file locs
    # of all finished, merged, permutations
    merged_vmax_files = glob.glob(
        get_merged_save_loc(temp_dr, '*', init_dr=False))

    # If not done, just return
    if len(merged_vmax_files) < n_perm:
        return

    # In the case that more than n_perm, be exact
    merged_vmax_files = merged_vmax_files[:n_perm]

    # Load and cast to array and save
    h0_vmax = np.array([np.load(file) for file in merged_vmax_files])
    np.save(os.path.join(results_dr, 'h0_vmax.npy'), h0_vmax)

    # Load the full original scores in order to compute ranks
    original_scores = np.load(os.path.join(results_dr, 'original_scores.npy'))
    if two_sided_test:
        original_scores = np.fabs(original_scores)

    # Compute scores as ranks by incrementing
    # every time the original score is >= vmax
    scores_as_ranks = np.zeros(original_scores.shape)
    for vmax in h0_vmax:
        scores_as_ranks += vmax < original_scores

    # Convert ranks into p-values and save
    pvals = (n_perm + 1 - scores_as_ranks) / float(1 + n_perm)
    np.save(os.path.join(results_dr, 'pvals.npy'), pvals)

    return

def select_next_fill_split(temp_dr):

    # Get list of not completed files
    temp_result_dr = os.path.join(temp_dr, 'temp_results')
    not_completed = get_not_completed(temp_dr)

    # Grab the current files info
    files = {}
    for rs in not_completed:
        try:
            files[rs] =\
                [int(file.split('.')[0]) for file in
                 os.listdir(os.path.join(temp_result_dr, rs))]
        except:
            continue

    # Set number of n_splits as highest found
    n_splits = 0
    for rs in files:
        highest = np.max(files[rs])
        if highest > n_splits:
            n_splits = highest
    
    # Init records
    records = np.zeros(n_splits+1)
    for rs in files:

        # Increment based on presence of each file
        for f in files[rs]:
            records[f] += 1

    # Select a new ind w/ prob based on sz
    biggest = np.max(records)

    # Initial probabilities are just max - number found
    probs = biggest - records

    # We want to prioritize the smaller ones to some extent more than linearly
    # setting higher values exponentially higher
    probs = probs ** 1.5

    # Normalize to between 0 and 1
    probs = probs / np.sum(probs)

    # Gen random state instance to select with
    rs = np.random.RandomState(seed=random.randint(0, 1000))
    choice = rs.choice(np.arange(len(records)), p=probs)
    return choice

def get_not_completed(temp_dr, arg_n=None):
    
    # Since this folder may not exist yet, wrap in a try/except 
    try:
        completed_files = os.listdir(os.path.join(temp_dr, 'results'))
    except:
        completed_files = []

    # Conv to set representing all completed files
    completed = set([r.split('.')[0] for r in completed_files])

    # Get all not completed files / random states
    temp_result_dr = os.path.join(temp_dr, 'temp_results')

    try:
        not_completed = set(os.listdir(temp_result_dr))
    except:
        not_completed = set([])

    # Remove completed from not completed
    not_completed = not_completed - completed

    # If passed arg_n limit to just not_completed specific to that arg_n
    if arg_n is not None:

        # For each random state in not completed
        # if arg_n done, remove that random state from
        # not completed
        for rs in list(not_completed):
            if os.path.exists(get_split_save_loc(temp_dr, rs, arg_n)):
                not_completed.remove(rs)

    return list(not_completed)
    

def next_random_state(random_state, temp_dr, is_filler, arg_n):

    # Base non filler case is just increment random state.
    if not is_filler:
        return random_state + 1

    # Make sure temp results exists
    os.makedirs(os.path.join(temp_dr, 'temp_results'), exist_ok=True)

    # Otherwise, if filler, check which runs are not finished yet,
    # and sleep while None
    while len(not_completed := get_not_completed(temp_dr, arg_n)) == 0:
        time.sleep(5)

    # Select randomly from the uncompleted jobs
    rs = int(random.choice(not_completed))

    return rs

def run_permutations(temp_dr, arg_n, time_est, job_id,
                     script_start_time, is_filler=False, n_run=0):

    # Load saved args - if data is split, loads just that chunk
    with open(os.path.join(temp_dr, f'args_{arg_n}.pkl'), 'rb') as f:
        args = pkl.load(f)
    print(f'Loaded: args_{arg_n}.pkl', flush=True)

    # Gen random state based on the passed rng + job_id if not filler
    if is_filler:
        random_state = next_random_state(None, temp_dr, is_filler, arg_n)
    else:
        random_state = args['rng'].randint(1, np.iinfo(np.int32).max - 1, size=job_id)[-1]
        print(f'Starting job w/ random_state: {random_state}, data_split {arg_n}.', flush=True)

    # Only run another permutation if limit - time elapsed is
    # more than the longest we expect a permutation to take
    while args['limit'] - (time.time() - script_start_time) > time_est + EXTRA_PAD:

        # Check to see if permutation limit already hit somewhere
        # and results already proc'ed so we should just quit
        check_already_finished(results_dr, temp_dr, n_run)

        # Want to check first to see if for some reason
        # this random state has already been run - increment random
        # state and restart loop
        if check_already_exists(temp_dr, arg_n, random_state):
            random_state = next_random_state(random_state, temp_dr, is_filler, arg_n)
            continue
        
        # Otherwise, start the permutation
        p_start_time = time.time()

        # Generate this permutation based on current random state
        p_set = _get_perm_matrix(args['permutation_structure'],
                                 random_state=random_state)

        # Get v or z stats for this permutation
        perm_scores = args['run_perm_func'](p_set=p_set, **args)

        # Convert perm scores to to absolute values if two sides
        if args['two_sided_test']:
            perm_scores = np.fabs(perm_scores)

        # Calc h0 vmax
        h0_vmax = np.nanmax(perm_scores)

        # Save this result, either for just a split
        # or merging data splits, ect...
        proc_perm_vmax(temp_dr, random_state, arg_n, h0_vmax, args['n_splits'])

        # Next random state
        random_state = next_random_state(random_state, temp_dr, is_filler, arg_n)

        # Increment n_run
        n_run += 1

        # If time from this run was longer than time_est, replace time_est
        p_time = time.time() - p_start_time
        if p_time > time_est:
            time_est = p_time

        # Verbose every
        if n_run % VERBOSE_EVERY == 0:
            print(f'Finished permutation {n_run} in: {p_time:.3f} (total time: {(time.time() - script_start_time):.3f})', flush=True)

        # Check end condition every
        if n_run % CHECK_END_EVERY == 0:
            check_end_condition(results_dr, temp_dr, args['n_perm'], args['two_sided_test'])

        # If a filler job, check every x, to select a new arg_n
        if is_filler and n_run % CHANGE_FILLER_EVERY == 0:
            
            # Select a new data split based on the slowest job so far
            new_arg_n = select_next_fill_split(temp_dr)

            # Restart the permutation loop if the data split is different
            # then the current one
            if int(new_arg_n) != int(arg_n):
                run_permutations(temp_dr=temp_dr,
                                 arg_n=new_arg_n,
                                 time_est=time_est,
                                 job_id=job_id, 
                                 script_start_time=script_start_time,
                                 is_filler=True, n_run=n_run)
                
                # Escape from the current function call in this case once done
                return n_run

    # Also include a final check outside of the loop
    check_end_condition(results_dr, temp_dr,
                        args['n_perm'], args['two_sided_test'])
    check_already_finished(results_dr, temp_dr, n_run)

    return n_run

# Un-pack args into vars
temp_dr, arg_n, time_est, job_id = unpack_args()
results_dr = Path(temp_dr).parent.absolute()

# If passed -1, then means this is a filler job
is_filler = False
if arg_n == -1:
    is_filler = True
    print('Starting filler job', flush=True)

    # Start by waiting one time est length
    time.sleep(time_est)

    # Then select an arg_n
    arg_n = select_next_fill_split(temp_dr)

# Run main loops
n_run = run_permutations(temp_dr, arg_n, time_est,
                         job_id, script_start_time,
                         is_filler=is_filler, n_run=0)

# Print if not quit earlier
end_job_print(n_run)
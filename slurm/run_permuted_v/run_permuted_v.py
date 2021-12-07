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
CHECK_END_EVERY = 5
CHANGE_FILLER_EVERY = 20
CHECK_PROGRESS_EVERY = 10

# Keep track of number start time for verbose / filler jobs
# as global variable
script_start_time = time.time()

def unpack_passed_args():

    # Unpack args
    args = list(sys.argv[1:])

    # Temp and results dr
    temp_dr = args[0]

    # The data dimension split
    data_split = int(args[1])

    # Estimate per job from submitting job
    time_est = float(args[2])

    # Job array id
    job_id = int(args[3])

    return temp_dr, data_split, time_est, job_id

def end_job_print(n_run, n_failed):

    print('Job Finished.', flush=True)
    print(f'Completed {n_run} permutations in {(time.time() - script_start_time):.3f} (n_failed={n_failed})')

def check_already_finished(temp_dr, n_run, n_failed):

    results_dr = Path(temp_dr).parent.absolute()
    
    # If h0_vmax is saved, then we know already done
    if os.path.exists(os.path.join(results_dr, 'h0_vmax.npy')):

        # If pvals done and temp_dr still exists
        # try to help clean up temp dr before exit
        if os.path.exists(os.path.join(results_dr, 'pvals.npy')):
            if os.path.exists(temp_dr):
                shutil.rmtree(temp_dr, ignore_errors=True)

        # Print before finish / quit
        end_job_print(n_run, n_failed)

        # Exit
        sys.exit()

def check_already_exists(temp_dr, data_split, random_state):

    # First check final save loc
    if os.path.exists(get_merged_save_loc(temp_dr, random_state)):
        return True

    # Next check the intermediate save_loc
    if os.path.exists(get_split_save_loc(temp_dr, random_state, data_split)):
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

def get_split_save_loc(temp_dr, random_state, data_split, init_dr=False):

    # Make sure the directory exists first if passed
    save_dr = os.path.join(temp_dr, 'temp_results', str(random_state))
    if init_dr:
        os.makedirs(save_dr, exist_ok=True)

    # Return save loc of this chunk of data's h0 vmax
    return os.path.join(save_dr, f'{data_split}.npy')

def proc_perm_vmax(temp_dr, random_state, data_split, h0_vmax, n_splits, n_run):

    # If n_splits is 1, means no data splits
    # were done, and we can save this result directly to merged_save_loc
    if n_splits == 1:
        
        merged_save_loc =\
            get_merged_save_loc(temp_dr, random_state, init_dr=True)
        np.save(merged_save_loc, h0_vmax)

        return n_run + 1

    # Otherwise, if there were splits then
    # check to see if saving this split
    # would finish across all data pieces
    split_save_locs = glob.glob(
        get_split_save_loc(temp_dr, random_state, data_split='*'))

    # The current split save loc
    split_save_loc =\
        get_split_save_loc(temp_dr, random_state, data_split, init_dr=True)

    # In this case, compute vmax across pieces
    if len(split_save_locs) == n_splits - 1:

        # There exists case where there may be n_splits - 1
        # but if the current split_save_loc already exists then
        # just return, as this is a duplicate case
        if os.path.exists(split_save_loc):
            return n_run

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
                return n_run

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

        return n_run + 1

    # If exists, return, marked as failed job
    if os.path.exists(split_save_loc):
        return n_run

    # Save split
    np.save(split_save_loc, h0_vmax)

    return n_run + 1

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

def select_next_fill_split(temp_dr, n_run, n_failed):

    # Get list of not completed files
    temp_result_dr = os.path.join(temp_dr, 'temp_results')
    not_completed = get_not_completed(temp_dr)

    # Grab the current files info
    files = {}
    for rs in not_completed:
        try:

            splits = [int(file.split('.')[0]) for file in
                      os.listdir(os.path.join(temp_result_dr, rs))]
            
            # Only add if some found
            if len(splits) > 0:
                files[rs] = splits

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

    # If sum == 0, wait then try again
    if np.sum(probs) == 0:

        # Check if already finished, to avoid getting stuck
        check_already_finished(temp_dr, n_run, n_failed)
        
        # Stall
        time.sleep(3)

        # Then try again
        return select_next_fill_split(temp_dr, n_run, n_failed)
        
    # Normalize to between 0 and 1
    probs = probs / np.sum(probs)

    # Gen random state instance to select with
    rs = np.random.RandomState(seed=random.randint(0, 1000))
    choice = rs.choice(np.arange(len(records)), p=probs)
    return choice

def get_completed_files(temp_dr):

     # Since this folder may not exist yet, wrap in a try/except 
    try:
        completed_files = os.listdir(os.path.join(temp_dr, 'results'))
    except:
        completed_files = []

    return completed_files

def get_not_completed(temp_dr, data_split=None):
    
    # Get completed files
    completed_files = get_completed_files(temp_dr)

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

    # If passed data_split limit to just not_completed specific to that data_split
    if data_split is not None:

        # For each random state in not completed
        # if data_split done, remove that random state from
        # not completed
        for rs in list(not_completed):
            if os.path.exists(get_split_save_loc(temp_dr, rs, data_split)):
                not_completed.remove(rs)

    return list(not_completed)
    
def next_random_state(random_state, temp_dr, is_filler, data_split):

    # Base non filler case is just increment random state.
    if not is_filler:
        return random_state + 1

    # Make sure temp results exists
    os.makedirs(os.path.join(temp_dr, 'temp_results'), exist_ok=True)

    # Get list of uncompleted jobs
    not_completed = get_not_completed(temp_dr, data_split)

    # If none, this is a good sign that this
    # data split is not a good option anymore. 
    if len(not_completed) == 0:
        return 'change'

    # Select randomly from the uncompleted jobs
    rs = int(random.choice(not_completed))

    return rs

def change_target_vars(temp_dr, is_filler, n_run, n_failed):
    '''This method is called when changing data split.
    Even if selects same as current, treat as low risk and just
    re-load anyway.'''

    # Select a new data split
    new_data_split = select_next_fill_split(temp_dr, n_run, n_failed)

    # Load new target vars split
    target_vars = load_target_vars(temp_dr, new_data_split)

    # Init with new random state
    random_state = next_random_state(None, temp_dr, is_filler, new_data_split)

    # If selects empty one, ... try again
    if random_state == 'change':
        del target_vars
        return change_target_vars(temp_dr, is_filler, n_run, n_failed)

    return target_vars, new_data_split, random_state

def load_target_vars(temp_dr, data_split):
    return np.load(os.path.join(temp_dr, f'target_vars_{data_split}.npy'))

def check_swap_filler(temp_dr, data_split, n_perm):

    # Get number of all complete
    base_done = len(get_completed_files(temp_dr))

    # Find number of saved in split loc w/ globbing
    split_done = len(glob.glob(get_split_save_loc(temp_dr, '*', data_split)))

    completed = base_done + split_done
    print(f'Completed {completed}/{n_perm} permutations in {(time.time() - script_start_time):.3f} (data split={data_split})', flush=True)
    
    # If over or equal to, then True
    if completed >= n_perm:
        return True
    
    # Otherwise False
    return False

def load_base_args(temp_dr):

    try:
        with open(os.path.join(temp_dr, 'base_args.pkl'), 'rb') as f:
            args = pkl.load(f)
    
    # If missing - check if permutations already done
    except FileNotFoundError:
        check_already_finished(temp_dr, 0, 0)

    return args

def run_permutations(temp_dr, results_dr, data_split, time_est, job_id,
                     script_start_time, is_filler=False):

    n_run, n_failed = 0, 0

    # Load base args - these don't change based on split
    args = load_base_args(temp_dr)

    # Load current split of target vars
    target_vars = load_target_vars(temp_dr, data_split)

    # Gen random state based on the passed rng + job_id if not filler
    if is_filler:
        random_state = next_random_state(None, temp_dr, is_filler, data_split)
        
        # Process possible data split change
        if random_state == 'change':
            del target_vars
            target_vars, data_split, random_state =\
                change_target_vars(temp_dr, is_filler, n_run, n_failed)

    else:
        random_state = args['rng'].randint(1, np.iinfo(np.int32).max - 1, size=job_id)[-1]
        print(f'Starting job w/ random_state: {random_state}, data_split {data_split}.', flush=True)

    # Only run another permutation if limit - time elapsed is
    # more than the longest we expect a permutation to take
    while args['limit'] - (time.time() - script_start_time) > time_est + EXTRA_PAD:

        try:

            # Check to see if permutation limit already hit somewhere
            # and results already proc'ed so we should just quit
            check_already_finished(temp_dr, n_run, n_failed)

            # Want to check first to see if for some reason
            # this random state has already been run - increment random
            # state and restart loop
            if check_already_exists(temp_dr, data_split, random_state):
                random_state = next_random_state(random_state, temp_dr, is_filler, data_split)

                # Process possible data split change
                if random_state == 'change':
                    del target_vars
                    target_vars, data_split, random_state = change_target_vars(temp_dr, is_filler, n_run, n_failed)

                # Skip rest of loop
                continue
            
            # Otherwise, start the permutation
            p_start_time = time.time()

            # Generate this permutation based on current random state
            p_set = _get_perm_matrix(args['permutation_structure'],
                                    random_state=random_state)

            # Get v or z stats for this permutation
            perm_scores = args['run_perm_func'](p_set=p_set, target_vars=target_vars, **args)

            # Convert perm scores to to absolute values if two sides
            if args['two_sided_test']:
                perm_scores = np.fabs(perm_scores)

            # Calc h0 vmax
            h0_vmax = np.nanmax(perm_scores)

            # Save this result, either for just a split
            # or merging data splits, ect...
            new_n_run = proc_perm_vmax(temp_dr, random_state,
                                       data_split, h0_vmax,
                                       args['n_splits'], n_run)

            # Get next random state
            random_state = next_random_state(random_state, temp_dr,
                                            is_filler, data_split)

            # If a filler job, check every x, or if random state condition
            if (is_filler and n_run % CHANGE_FILLER_EVERY == 0) or (random_state == 'change'):
                del target_vars
                target_vars, data_split, random_state = change_target_vars(temp_dr, is_filler, n_run, n_failed)

            # If time from this run was longer than time_est, replace time_est
            p_time = time.time() - p_start_time
            if p_time > time_est:
                time_est = p_time

            # Check if was a failed job,
            # in that case, skip rest of loop
            # this is just to avoid printing progress
            # or whatever if run was a failure
            if new_n_run == n_run:
                n_failed += 1
                continue
            
            # Set to new
            n_run = new_n_run

            # Check end condition every
            if n_run % CHECK_END_EVERY == 0:
                check_end_condition(results_dr, temp_dr, args['n_perm'], args['two_sided_test'])

            # Update progress every
            if n_run % CHECK_PROGRESS_EVERY == 0:
                
                # Check to see if this data split is at the requested number
                # of permutations. In that case, switch it to a filler job
                # instead of running more permutations then needed
                if not is_filler:

                    if check_swap_filler(temp_dr, data_split, args['n_perm']):
                        print('Base permutations finished, switching to filler job', flush=True)

                        # Set to filler
                        is_filler = True

                        # Change target vars
                        del target_vars
                        target_vars, data_split, random_state = change_target_vars(temp_dr, is_filler, n_run, n_failed)

                # Filler-case, just print
                else:
                    print(f'This job has completed {n_run} total permutations in {(time.time() - script_start_time):.3f}', flush=True)


        # If missing ever hit file not found, check to see, could be already finished
        except FileNotFoundError:
            check_already_finished(temp_dr, n_run, n_failed)

    # Also include a final check outside of the loop
    check_end_condition(results_dr, temp_dr,
                        args['n_perm'], args['two_sided_test'])
    check_already_finished(temp_dr, n_run, n_failed)

    return n_run, n_failed

def main():

    # Un-pack args into vars
    temp_dr, data_split, time_est, job_id = unpack_passed_args()
    results_dr = Path(temp_dr).parent.absolute()

    # If passed -1, then means this is a filler job
    is_filler = False
    if data_split == -1:
        is_filler = True
        print('Starting filler job', flush=True)

        # Start by waiting one time est length
        time.sleep(time_est)

        # Then select an data_split
        data_split = select_next_fill_split(temp_dr, 0, 0)

    # Run main loops
    n_run, n_failed = run_permutations(temp_dr, results_dr, data_split,
                                       time_est, job_id, script_start_time,
                                       is_filler=is_filler)

    # Print if not quit earlier
    end_job_print(n_run, n_failed)


if __name__ == '__main__':
    main()


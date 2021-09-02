from funcs import get_setup, single_run
from config import get_job_limit
import os
import random
import time
import numpy as np
import pickle as pkl

#### User defined variables / pieces ####

# Location where csv version of rds is saved
rds_loc = '/users/s/a/sahahn/Parcs_Project/raw/nda_rds_201.csv'

# The eventname in which to filter results by
eventname = 'baseline_year_1_arm_1'

# Save loc / name for folder where results will be saved
# Note: If this folder already exists then the previous results will be overwritten
results_loc = 'my_results1'

# The random effects to use in the model, where if
# using two, then first is the outer group e.g., site,
# and the second is the inner effect e.g., family
# Note that variable names can't contain a '.' for now.
random_effects_vars = ['abcd_site', 'rel_family_id']

# The fixed effects variable names
# where if any are categorical they should be wrapped in 'C(variable_name)'
# Note that variable names can't contain a '.' for now.
fixed_effects_vars = ['nihtbx_picvocab_uncorrected', 'interview_age']

# The template path used when loading the data where CONTRAST and SUBJECT are
# replaced by contrast and each subject's loaded index name respectively.
# This is used to provide a way of mapping subjects to their saved file location.
template_path = '/users/s/a/sahahn/ABCD_Data/All_Merged_Vertex_2.0.1/CONTRAST/SUBJECT_lh.mgz'

# The contrast used in template_path above
# Note this can be set to None and then CONTRAST removed from
# above also, as it is optional.
contrast = 'correct_go_vs_fixation'

# Use short jobs (3hr limit) or bluemoon jobs if False (30hr limit)
# tradeoff is will need less of bluemoon jobs but they will take longer.
use_short_jobs = True


#### Code starts Below ####

start_time = time.time()

# Make sure directories exists
os.makedirs('Job_Logs', exist_ok=True)
os.makedirs(results_loc, exist_ok=True)

# Load in rds with relevant cols - no NaN's
df = 


# Get rds with relevant cols - no NaN's
df = get_setup(indep_var, fixed_effects_vars,
            random_effects_vars, rds_loc, eventname)

# Loads subjects as NDAR_ style, change to just base
df.index = [i.replace('NDAR_', '') for i in df.index]

# Get all subjects and setup
subjects, _print = setup_subjects(df, template_path, contrast, n_jobs, verbose=0)

# Re-index base df
df = df.loc[subjects]

# Load in the imaging data
data = get_data(subjects, contrast, template_path, mask=None,
                index_slice=None, n_jobs=n_jobs, _print=_print)

print('Loaded:', data.shape, df.shape, 'in',
       time.time() - start_time,
       'seconds.', flush=True)

# Init temp directory - should not already exist
temp_dr = 'temp_' + str(random.random())
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
        'indep_var': indep_var,
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
submit_cmd = f'sbatch --array=1-{int(jobs_needed)} {script_name} {temp_dr}'
os.system(submit_cmd)

print(f'Submitted {int(jobs_needed)} jobs.')
print('If this is not enough and more jobs need to be run, you can submit')
print('more jobs by modifying the following command (change 20 to number of extra jobs):')
print(submit_cmd.replace(str(int(jobs_needed)), '20'))

print()
print(f'Final results when finished will be saved in directory: {results_loc}')

print()
print('In the case where jobs start failing with memory errors,')
print('you must manually change the line in files run_mm.sh and/or')
print('run_short_mm.sh "#SBATCH --mem=1G" to a higher value.')

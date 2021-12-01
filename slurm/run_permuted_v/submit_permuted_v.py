from math import perm
from neurotools.loading.abcd import load_from_csv, load_family_block_structure
from neurotools.loading.funcs import get_overlap_subjects, load_data
from neurotools.loading.outliers import drop_top_x_outliers
from funcs import setup_and_run_permuted_v
import pandas as pd
import numpy as np
import sys

# Save dr / name for folder where results will be saved
results_dr = 'temp'

# loc of CSV with all needed columns
csv_loc = '/users/s/a/sahahn/ABCD_Data/nda3.0.csv'

# The eventname in which to filter results by
eventname = 'baseline_year_1_arm_1'

# The variable to test - no support for multiple here (yet)
tested_vars = ['neurocog_pc1.bl']

# The confounding vars
# where if any are categorical they should be wrapped in 'C(variable_name)'
# these will be dummy coded
confounding_vars = ['C(sex_at_birth)',
                    'C(abcd_site)',
                    'C(race_ethnicity)',
                    'interview_age']

# The template path used when loading the data where SUBJECT is replaced
# by the passed loaded subject index name
# This is used to provide a way of mapping subjects to their saved file location.
template_path = '/users/s/a/sahahn/ABCD_Data/All_Merged_Voxel_2.0.1/nBack_2_back_vs_0_back/SUBJECT_nBack.nii.gz'

# Apply mask to data
mask = '/users/s/a/sahahn/ABCD_Data/sub_mask.nii'

# Optionally, can drop the top X number of outliers
# just determined crudely by absolute mean value
drop_top = 200

#### Options for how permutations are run / submitted ####

# The number of permutations to use
n_perm = 500

# Use short jobs (3hr limit) or bluemoon jobs if False (30hr limit)
# tradeoff is will need less of bluemoon jobs but they will take longer.
use_short_jobs = True

# Grab the job memory used to submit this job
# and specify that same value be used, or override here
job_mem = sys.argv[1]

# The number of jobs to submit is based on
# an estimate of how long on average each permutation
# will take, increasing this job_factor multiplies
# the estimated time, e.g., a higher number will
# submit more jobs and therefore finish faster
# where a smaller job factor close to 1, might not
# even finish the requested number of permutations.
job_factor = 5

# Imaging data can be split to run
# permutations seperately per chunks of data
# which is helpful to reduce the computational
# intensity / requirements of each submitted job.
# This parameter specifies the target size (number of vertex or voxel)
# to run at once. Data will be split such that each chunk is as close to this size
# as possible.
sz_target = 2000

# Optionally de-mean confounds
demean_confounds = True

results_dr = f'test_{n_perm}_{job_factor}_{sz_target}'

#### Code starts Below ####

# Get all cols
all_vars = tested_vars + confounding_vars

# Load target + confounds, dummy coded + no nans
df = load_from_csv(cols=all_vars,
                   csv_loc=csv_loc,
                   eventname=eventname,
                   drop_nan=True,
                   encode_cat_as='dummy',
                   verbose=1)
print('Loaded initial df', df.shape, flush=True)

# Update confounding vars to reflect names after dummy coding
confounding_vars = list(df)
confounding_vars.remove(tested_vars[0])

# Change subject index if needed to match saved file names
df.index = [i.replace('NDAR_', '') for i in df.index]

# Get overlap of subjects with df and imaging data
subjects = get_overlap_subjects(df=df,
                                template_path=template_path,
                                verbose=1)

# Load default permutation structure
subjects = ['NDAR_' + s for s in subjects]
permutation_structure = load_family_block_structure(
                            csv_loc=csv_loc, 
                            subjects=subjects,
                            eventname=eventname,
                            verbose=1)

# Change subject index if needed to match saved file names
permutation_structure.index = [i.replace('NDAR_', '') for i in permutation_structure.index]

# Get latest overlap of subjects as permutation structure df index
subjects = permutation_structure.index

# Load in the imaging data
target_vars = load_data(subjects=subjects,
                        template_path=template_path,
                        mask=mask,
                        nan_as_zero=True,
                        n_jobs=4, verbose=1)

# Perform optional outlier filtering
target_vars, subjects = drop_top_x_outliers(target_vars, subjects, drop_top)
print(f'Loaded Max-Min after drop: {drop_top} outliers {np.max(target_vars)} {np.min(target_vars)}', flush=True)
print(flush=True)

# Now perform final update to loaded subjects in df and permutation_structure to match
# subjects after outlier filtering.
df = df.loc[subjects]
permutation_structure = permutation_structure.loc[subjects]

# Submit jobs - casting df's to numpy arrays where needed
setup_and_run_permuted_v(results_dr,
                         tested_vars=np.array(df[tested_vars]),
                         target_vars=target_vars,
                         confounding_vars=np.array(df[confounding_vars]),
                         permutation_structure=np.array(permutation_structure),
                         n_perm=n_perm,
                         two_sided_test=True,
                         within_grp=True,
                         random_state=None,
                         use_tf=False,
                         dtype=None,
                         use_z=False,
                         demean_confounds=demean_confounds,
                         use_short_jobs=use_short_jobs,
                         job_mem=job_mem,
                         job_factor=job_factor,
                         sz_target=sz_target)


                         
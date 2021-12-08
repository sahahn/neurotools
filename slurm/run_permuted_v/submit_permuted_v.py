from funcs import setup_and_run_permuted_v, load_and_setup_data
from nilearn.masking import compute_brain_mask
import numpy as np
import nibabel as nib

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

# The template path used when loading the data and can be either a str template or function
def template_path(subject, contrast=None):
    sub = subject.replace('NDAR_', '')
    return f'/users/s/a/sahahn/ABCD_Data/All_Merged_Voxel_2.0.1/nBack_2_back_vs_0_back/{sub}_nBack.nii.gz'

# Compute mask
# mask = '/users/s/a/sahahn/ABCD_Data/sub_mask.nii'
ex_data = nib.load(template_path('NDAR_INV92V7K5PE'))
mask = compute_brain_mask(ex_data, threshold=.2).get_fdata()

# Optionally, can drop the top X number of outliers
# just determined crudely by absolute mean value
drop_top = 1000

#### Options for how permutations are run / submitted ####

# The number of permutations to use
n_perm = 100

# Use short jobs (3hr limit) or bluemoon jobs if False (30hr limit)
# tradeoff is will need less of bluemoon jobs but they will take longer.
use_short_jobs = True

# Use default job mem settings
# or override specifically here
job_mem = None

# The number of jobs to submit is based on
# an estimate of how long on average each permutation
# will take, increasing this job_factor multiplies
# the estimated time, e.g., a higher number will
# submit more jobs and therefore finish faster
# where a smaller job factor close to 1, might not
# even finish the requested number of permutations.
# So in theory, if submitting short jobs with a max time of 3hrs,
# and using a job_factor of 10, the permutation jobs should
# finish in around 3 hrs / 10, so 18 minutes.
# Note that this doesn't include the time it takes the submitting
# job to load and calculate the base scores.
job_factor = 10

# Imaging data can be split to run
# permutations seperately per chunks of data
# which is helpful to reduce the computational
# intensity / requirements of each submitted job.
# This parameter specifies the target size (number of vertex or voxel)
# to run at once. Data will be split such
# that each chunk is as close to this size as possible.
sz_target = 50000

# Optionally de-mean confounds
demean_confounds = True

# Set to None to use whatever loaded data is,
# but float32 produces very simmilar results
# and is less memory intensive.
dtype = 'float32'

# The maximum number of jobs a single permutation can submit
job_limit = 500

# The minimum variance group size allowed
min_vg_size =  5

####

results_dr = f'test_{n_perm}_{job_factor}_{sz_target}'

#### Code starts Below ####

# Load and setup data
tested_vars, target_vars, confounding_vars, permutation_structure =\
    load_and_setup_data(tested_vars, confounding_vars, csv_loc,
                        eventname, template_path, mask, dtype, drop_top)

# Submit jobs
setup_and_run_permuted_v(results_dr, tested_vars=tested_vars,
                         target_vars=target_vars, confounding_vars=confounding_vars,
                         permutation_structure=permutation_structure, n_perm=n_perm,
                         two_sided_test=True, within_grp=True, random_state=None,
                         use_tf=False, dtype=dtype, use_z=False, demean_confounds=demean_confounds,
                         use_short_jobs=use_short_jobs, job_mem=job_mem, job_factor=job_factor,
                         sz_target=sz_target, job_limit=job_limit, min_vg_size=min_vg_size)


                         
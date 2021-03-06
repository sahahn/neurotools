from funcs import run_slurm_mm
from neurotools.loading import get_overlap_subjects, load_data
from neurotools.loading.abcd import load_from_csv
from neurotools.misc.text import name_replace
import pandas as pd

#### User defined variables / pieces ####

# Location where csv version of rds is saved
rds_loc = '/users/s/a/sahahn/Parcs_Project/raw/nda_rds_201.csv'

# The eventname in which to filter results by
eventname = 'baseline_year_1_arm_1'

# Save loc / name for folder where results will be saved
# Note: If this folder already exists then the previous results will be overwritten
results_loc = 'my_results1'

# The fixed effects variable names
# where if any are categorical they should be wrapped in 'C(variable_name)'
fixed_effects_vars = ['neurocog_pc1.bl', 'C(sex_at_birth)',
                      'interview_age', 'C(abcd_site)',
                      'C(race_ethnicity)']

# The random effects to use in the model, where if
# using two, then first is the outer group e.g., site,
# and the second is the inner effect e.g., family
random_effects_vars = ['rel_family_id']

# The template path used when loading the data where CONTRAST and SUBJECT are
# replaced by contrast and each subject's loaded index name respectively.
# This is used to provide a way of mapping subjects to their saved file location.
template_path = '/users/s/a/sahahn/ABCD_Data/CONTRAST/SUBJECT.npy'

# The contrast used in template_path above
# Note this can be set to None and then CONTRAST removed from
# above also, as it is optional.
contrast = '2_back_vs_0_back_concat'

# Use short jobs (3hr limit) or bluemoon jobs if False (30hr limit)
# tradeoff is will need less of bluemoon jobs but they will take longer.
use_short_jobs = True

# Optionally, pass location of files with the name
# of one subject per row, to filter subjects by, i.e., only
# subjects that exist in this file will be used.
qc_subjs_loc = '/users/s/a/sahahn/ABCD_Data/nback_valid_subjs.txt'


#### Code starts Below ####
# Essentially this whole file is just preparing arguments
# for the function run_slurm_mm

# Load in rds with relevant cols - no NaN's
all_vars = fixed_effects_vars + random_effects_vars

df = load_from_csv(cols=all_vars,
                   rds_loc=rds_loc,
                   eventname=eventname,
                   drop_nan=True)

# Load in set of qc'ed subjects, then set overlap
if qc_subjs_loc:
    qc_valid_subjs = pd.Index(pd.read_csv(qc_subjs_loc, header=None)[0])
    overlap = df.index.intersection(qc_valid_subjs)
    df = df.loc[overlap]

# Replace periods in both loaded df and var names
df = name_replace(df, '.', '')
fixed_effects_vars = name_replace(fixed_effects_vars, '.', '')
random_effects_vars = name_replace(random_effects_vars, '.', '')

# Loads subjects as NDAR_ style, change to just base
df.index = [i.replace('NDAR_', '') for i in df.index]

# Get overlap of subjects
subjects = get_overlap_subjects(subjs=df, template_path=template_path,
                                contrast=contrast, verbose=1)

# Re-index base df
df = df.loc[subjects]

# Load in the imaging data
data = load_data(subjects=subjects, template_path=template_path,
                contrast=contrast, mask=None, index_slice=None,
                zero_as_nan=True,
                n_jobs=1, verbose=1)

# Submit job to run on slurm cluster multi-proc
run_slurm_mm(df=df, data=data,
             fixed_effects_vars=fixed_effects_vars,
             random_effects_vars=random_effects_vars,
             results_loc=results_loc,
             use_short_jobs=use_short_jobs,
             job_mem='1G')

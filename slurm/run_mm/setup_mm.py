from funcs import run_slurm_mm
from neurotools.loading import get_overlap_subjects, load_from_abcd_rds, get_data

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
# Essentially this whole file is just preparing arguments
# for the function run_slurm_mm

# Load in rds with relevant cols - no NaN's
df = load_from_abcd_rds(cols=fixed_effects_vars + random_effects_vars,
                        eventname='baseline_year_1_arm_1',
                        drop_nan=True)

# Loads subjects as NDAR_ style, change to just base
df.index = [i.replace('NDAR_', '') for i in df.index]

# Get overlap of subjects
subjects = get_overlap_subjects(df=df, template_path=template_path,
                                contrast=contrast, verbose=1)

# Optionally here filter subjects further, e.g,
# subjects = list(set(subjects) - set(invalid_subjects))

# Re-index base df
df = df.loc[subjects]

# Load in the imaging data
data = get_data(subjects=subjects, template_path=template_path,
                contrast=contrast, mask=None, index_slice=None,
                n_jobs=1, verbose=1)

# Submit job to run on slurm cluster multi-proc
run_slurm_mm(df=df, data=data,
             fixed_effects_vars=fixed_effects_vars,
             random_effects_vars=random_effects_vars,
             results_loc=results_loc,
             use_short_jobs=use_short_jobs)

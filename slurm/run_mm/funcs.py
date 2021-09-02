

def get_setup(indep_var, fixed_effects_vars, random_effects_vars, rds_loc, eventname):

    # Load all data
    cols = random_effects_vars + [indep_var] + fixed_effects_vars
    data = load_from_rds(rds_loc=rds_loc, col_names=cols, eventname=eventname)

    # Drop NaN's
    data.dropna(inplace=True)

    # Ordinal encode random effects
    oe = OrdinalEncoder()
    data[random_effects_vars] = oe.fit_transform(data[random_effects_vars])

    return data
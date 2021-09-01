from sklearn.preprocessing import OrdinalEncoder
import statsmodels.api as sm

def run_mixed_model(df, fixed_effects_vars, random_effects_vars):
    '''Wrapper code for running a mixed linear model with either one random effect
    or two nested random effects.

    Parameters
    -----------
    df : pandas DataFrame
        A DataFrame containing columns for the fixed effects
        and the random effects.

        The dataframe should also contain a single column with
        name 'data' which represents the dependent variable,
        e.g., `data ~ fixed_effects_vars`

    fixed_effects_vars : list of str
        List of fixed effect variables, where if any
        variables are categorical they should be wrapped in `C(variable_name)`.

    random_effects_Vars : list of str or str
        Either a single random effect or a list with two
        random effect variables specifying that multiple
        layers of nesting should be done. In the case
        where two variables should be nested, the first element
        of the list represents the larger group and the second
        the nested group within the larger group.

        As of right now only these two options are supported.
    
    Returns
    -------
    fitted_model : MixedLMResults
        Returns an instance of MixedLMResults with fitted results.

    '''
    
    # If just one random effect specified
    if len(random_effects_vars) == 1:
        if isinstance(random_effects_vars, str):
            random_effects_vars = [random_effects_vars]
        
        re_formula, vc_formula = None, None
    
    # Specify nested structure if two random effects
    elif len(random_effects_vars) == 2:
        
        re_formula = "1"
        vc_formula = {random_effects_vars[1]: "0+C(" + random_effects_vars[1] + ")"}

    else:
         raise RuntimeError('More than 2 random effects are not currently supported.')
    
    # Make sure random effects are ordinalized
    df[random_effects_vars] = OrdinalEncoder().fit_transform(df[random_effects_vars])
    
    # Set base formula as data explained by fixed effect vars
    base = 'data ~ ' + fixed_effects_vars[0]
    base += ' + '.join([''] + fixed_effects_vars[1:])
    
    # Get model from formula
    model =\
        sm.MixedLM.from_formula(base, groups=random_effects_vars[0],
                                re_formula=re_formula,
                                vc_formula=vc_formula, data=df)
    result = model.fit()
    return result
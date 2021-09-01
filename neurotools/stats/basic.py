from sklearn.linear_model import LinearRegression
import numpy as np

def get_resid(covars, data):
    '''Get residualized data with two function versions, with or without
    missing data.'''

    if np.isnan(data).any():
        return _get_resid_with_nans(covars, data)
    return _get_resid_without_nans(covars, data)

def _get_resid_with_nans(covars, data):

    # Go from pandas df to numpy array
    covars = np.array(covars)

    # Make sure data is numpy array
    data = np.array(data)
    
    # Init empty resid array of NaN's
    resid = np.empty(shape=data.shape)
    resid[:] = np.nan
    
    # For each feature seperately
    for i in range(data.shape[1]):
        
        # Operate on non-nan subjects for this feature
        mask = ~np.isnan(data[:, i])
        
        # If not at least 2 subjects valid,
        # skip and propegate NaN's for this
        # voxel.
        if np.sum(mask) > 1:
        
            # Fit model
            model = LinearRegression().fit(covars[mask], data[mask, i])
            
            # Compute difference of real value - predicted
            dif_i = data[mask, i] - model.predict(covars[mask])
            
            # Set resid as diff + intercept
            resid_i = model.intercept_ + dif_i
            
            # Fill in NaN mask
            resid[mask, i] = resid_i
    
    return resid

def _get_resid_without_nans(covars, data):

    # Go from pandas df to numpy array
    covars = np.array(covars)

    # Make sure data is numpy array
    data = np.array(data)

    # Fit a linear regression with covars as predictors, each voxel as target variable
    model = LinearRegression().fit(covars, data)

    # The difference is the real value of the voxel, minus the predicted value
    dif = data - model.predict(covars)

    # Set the residualized data to be, the intercept of the model + the difference
    resid = model.intercept_ + dif

    return resid

def get_cohens(data):
    '''Get cohen's d value with or without any
    missing values present in data'''

    if np.isnan(data).any():
        return _get_cohens_with_nan(data)
    return _get_cohens_without_nans(data)

def _get_cohens_with_nan(data):

    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    cohen = mean / std

    return cohen

def _get_cohens_without_nans(data):

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    cohen = mean / std

    return cohen

def fast_corr(O, P):
    '''Compute a very fast correlation coef.'''
    
    n = P.size
    DO = O - (np.sum(O, 0) / np.double(n))
    DP = P - (np.sum(P) / np.double(n))
    return np.dot(DP, DO) / np.sqrt(np.sum(DO ** 2, 0) * np.sum(DP ** 2))
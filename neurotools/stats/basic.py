from sklearn.linear_model import LinearRegression
import numpy as np

def get_resid(covars, data):
    '''Compute the simple linear residualized version of a set of data according
    to a passed set of covariates. This fits a linear regression from covariates to data
    and then returns the difference between the predicted values and the true values for
    the data.

    In this case of missing data, within the data portion of the input,
    this method will operate on only the the subset of subject's with
    non-missing data per feature, any NaN data will be propegated to
    the returned residualized data.

    Note that the intercept information from the linear model
    is re-added to the residuals. 

    Parameters
    ------------
    covars : numpy array
        The covariates in which to use to residualize
        the passed data, seperate per feature. This
        input must have shape number of subjects x number of
        covariates.

    data : numpy array
        The data in which to residualize. This
        input muct have shape number of subjects x number of
        data features

    Returns
    ---------
    resid_data : numpy array
        A residualized version, with the same shape,
        of the passed data is returned. Any NaN values
        in the original input data are preserved.
    '''

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
            
            # Compute difference of real value - predicted, then add intercept
            resid_i = data[mask, i] - model.predict(covars[mask]) + model.intercept_
            
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

    # The difference is the real value of the voxel, minus the predicted value, + intercepts
    resid = data - model.predict(covars) + model.intercept_

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
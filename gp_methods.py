
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.optimize import minimize
import pdb
from scipy.stats import multivariate_normal

def squared_exponential(x, b, tau_sq1, tau_sq2=10**-6):

    """

    The squared exponential case of the matern class

    Parameters
    ----------
    x : array_like
        The points at which to evaluate the function
    b : float
        the first hyperparameter
    tau_sq1: float
        the second hype parameter
    tau_sq2: float
        the third hyperparameter

    Returns
    -------
    cov_mat : array_like
        The covariance matrix based on a squared exponential kernel

    """

    #Need to deal with 1-D arrays differently
    if x.ndim == 1:
        x_calc = [[i] for i in x]
    else:
        x_calc = x.copy()


    d_mat = distance_matrix(x_calc,x_calc)
    cov_mat = tau_sq1*np.exp(-.5*(d_mat/b)**2) + tau_sq2*np.eye(len(x_calc))

    return cov_mat


def gp_predict(x_data, y_data, x_pred,sigma_2, b,tau_sq1, tau_sq2, prior_mean = 0.0, cov_fun=squared_exponential):
    """

    Returns predictions of the mean of the function distribution for a
    Gaussian process

    Parameters
    ----------
    x_data : array_like
        the x values from the data
    y_data : array_like
        the corresponding y values from the data
    x_pred : array_like
        the values at which predictions will be made
    sigma_2 : float
        the variance of the residuals
    b: float
        hyperparameter for the covariance function
    tau_sq1: float
        hyperparameter for the covariance function
    tau_sq2: float
        hyperparameter for the covariance function
    prior_mean: float
        the mean value for the gaussian process (becomes vectorized)
    cov_fun: function
        the function to use to generate the covariance matrix

    Returns
    -------
    y_pred: array_like
        the predicted y values that correspond to x_pred
    cov: array_like
        the covariance matrix for the estimate of f(x_star)
    """
    C = cov_fun(np.concatenate([x_data, x_pred]), b, tau_sq1, tau_sq2)
    #Then we need to extract the partioned matrices
    #First C(x,x) =  C11
    C_11 = C[:len(x_data), :len(x_data)]
    C_21 = C[len(x_data):,:len(x_data)]
    C_22 = C[len(x_data):,len(x_data):]
    #then calculate the weight matrix
    w=C_21@np.linalg.inv((C_11 + np.eye(len(x_data))*sigma_2))
    #finally calculate the predicted y values
    y_pred = w@y_data
    cov=C_22 - C_21@np.linalg.inv(C_11+np.eye(len(x_data))*sigma_2)@np.transpose(C_21)
    return y_pred, cov

def log_likelihood(x_data, y_data, sigma_2, b, tau_sq1, tau_sq2 = 10**-6, cov_fun=squared_exponential):
    """
    Returns a quantity that is proportional to the log likelihood for a Gaussian process
    Used to determine hyperparameters for the matern covariance functions

    Parameters
    ----------
    x_data: array_like
        x values from the data
    y_data: array_like
        corresponding y values
    sigma_2: float
        Variance of residuals
    b: float
        Hyperparameter
    tau_sq1: float
        Hyperparameter
    tau_sq2: float
        Hyperparameter
    cov_fun: function
        The covariance function to use
    Returns
    -------
    log_like: float
        Quantity proportional to the log likelihood

    """


    #First evaluate the covariance matrix:
    C = cov_fun(x_data,b,tau_sq1, tau_sq2)
    p = multivariate_normal.logpdf(y_data, np.zeros(len(y_data)), sigma_2*np.eye(len(y_data))+C)
    return p

import numpy as np 
from scipy.stats import norm

from numpy import random


name = 'Tobit'

# criterion function
def q(theta, y, x): 
    return None # Fill in 

def loglikelihood(theta, y, x): 
    assert True # FILL IN: add some assertions to make sure that dimensions are as you assume 

    # unpack parameters 
    b = theta[:-1] # first K parameters are betas, the last is sigma 
    sig = np.abs(theta[-1]) # take abs() to ensure positivity (in case the optimizer decides to try it)

    ll = None # fill in 

    return ll

def starting_values(y,x): 
    '''starting_values
    Returns
        theta: K+1 array, where theta[:K] are betas, and theta[-1] is sigma (not squared)
    '''
    N, K = x.shape

    b_ols = np.linalg.solve(x.T@x, x.T@y) # OLS estimates as starting values
    
    residuals = y - x@b_ols 
    sigma2_hat = 1/(N-K) * np.dot(residuals, residuals)
    sigma_hat = np.sqrt(sigma2_hat) # OLS estimate of sigma = sqrt(sigma^2) as starting value 
    theta_0 = np.append(b_ols, sigma_hat)

    return theta_0 

def predict(theta, x): 
    '''predict(): the expected value of y given x 
    Returns E, E_pos
        E: E(y|x)
        E_pos: E(y|x, y>0) 
    '''
    # Fill in 
    b = theta[:-1]
    sig = theta[-1]

    # Input to mills_ratio:
    z = x@b / sig
    mills_ratio = norm.pdf(z) / norm.cdf(z)

    E = x@b * norm.cdf(x@b/sig) + sig * norm.pdf(x@b/sig)
    Epos = x@b + sig * mills_ratio
    return E, Epos

def sim_data(theta, N:int): 
    b = theta[:-1]
    sig = theta[-1]
    K = b.size

    # FILL IN 
    # x = random.normal(size=(N, b.shape[0])) 
    # x[:,0]=np.ones((N))
    xx = np.random.normal(size=(N,K-1))
    oo = np.ones((N,1))
    x  = np.hstack([oo,xx])

    u = np.random.normal(loc=0, scale=sig, size=(N,)) 
    ys = x @ b + u 
    y = np.maximum(ys, 0.0)

    return y, x

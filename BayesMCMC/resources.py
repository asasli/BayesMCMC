import numpy as np
from scipy.integrate import quad
import scipy.stats as st

def func_pdf(alpha, dist_max, dist):
    """The probability density function at 'x' given the parameters 'alpha' and 'x_max'."""
    
    """dist_max: maximum distance """
    """dist : distances"""
    
    # This is the same distance prior as above, but with an arbitrary exponent, alpha
    P_x = (alpha + 1.0) * dist ** alpha / dist_max ** (alpha + 1.0)
    
    # If distance is less than zero or greater than the maximum distance, the likelihood is zero
    if isinstance(dist, np.ndarray):
        P_x[dist > dist_max] = 0.0 
        P_x[dist < 0.0] = 0.0
    elif (dist > dist_max) or (dist < 0.0):
        P_x = 0.0
    
    return P_x

def func_integrand(dist, alpha, dist_max, plx_obs, plx_err):
    """ Calculate the integrand in the marginalization written above """
    
    return st.norm.pdf(1.0/dist, loc=plx_obs, scale=plx_err) * func_pdf(alpha, dist_max, dist)


def ln_prior(alpha):
    """The log-prior on 'alpha'. Prior distribution on alpha """
    if alpha <= 0.0:
        return -np.inf
    return 0.0

def ln_posterior(alpha, dist_max, plx_obs, plx_err):
    """ Log of the posterior function is the sum of the log of the prior and likelihood functions """

    """The log-posterior given the data (x_obs, x_err) and the model parameter."""
    return ln_prior(alpha) + ln_likelihood(alpha, dist_max, plx_obs, plx_err)

def ln_likelihood(alpha, dist_max, plx_obs, plx_err):
    """ Likelihood function requires the integration of a function """
    
    result = 0.0
    
    # Cycle through each observed star
    for i in range(len(plx_obs)):
        
        # Limits are either the limits of the pdf or 5 sigma from the observed value
        a = max(0.0, 1.0/(plx_obs[i] + 5.0 * plx_err[i]))
        b = min(dist_max, 1.0/(np.max([1.0e-5, plx_obs[i] - 5.0 * plx_err[i]])))

        # Calculate the integral
        val = quad(func_integrand, a, b, 
                   args=(alpha, dist_max, plx_obs[i], plx_err[i]), 
                   epsrel=1.0e-4, epsabs=1.0e-4)

        # Add the log likelihood to the overall sum
        result += np.log(val[0])

    return result

def metro_hastings(ln_posterior, theta_0, N_steps, step_size=0.2, args=[]):
    """Metropolis-Hastings algorith for sampling the posterior distribution.

    Parameters
    ----------
    ln_posterior : function that returns the logarithm of the posterior
    theta_0      : initial guess for the model parameter
    N_steps      : the length of the Markov Chain that will be returned
    step_size    : the standard deviation of the normally distributed step size
    args         : additional arguments to be passed to the posterior function
    
    Returns
    -------
    A numpy array containing the Markov Chain.
    
    """
    chain = np.zeros(N_steps)                     # create the chain
    chain[0] = theta_0                            # store the initial point...
    print("{:.3f}".format(chain[0]), end=",")     # ...and print it!
    
    # hold the current value of the posterior to avoid recomputing it if position is not changed
    curr_P = ln_posterior(theta_0, *args)
    
    # populate the rest of the point in the chain
    for i in range(N_steps - 1):
        new_theta = chain[i] + np.random.normal(scale=step_size)
        new_P = ln_posterior(new_theta, *args)
        
        # should we move to the new position?
        if (new_P > curr_P) or (np.random.rand() < np.exp(new_P - curr_P)):
            # if yes... store the new value, print it and update the 'current posterior'
            chain[i + 1] = new_theta
            print("{:.3f}".format(chain[i + 1]), end=",")
            curr_P = new_P
        else:
            # if no... store again the current position and print a '.'
            chain[i + 1] = chain[i]
            print(".", end=", ")
            
    return chain
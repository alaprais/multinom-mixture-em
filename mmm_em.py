import numpy as np
import math
from numba import njit


def multinom_mix_em_init(k: int, p: int) -> tuple[np.array, np.array]:
    multinom_probs = np.random.rand(k, p)
    multinom_probs /= np.sum(multinom_probs, axis=1)[:, np.newaxis]
    multinom_probs = np.clip(multinom_probs, a_min = 1e-100, a_max = 1 - 1e-100)
    
    mixture_weights = np.random.rand(k)
    mixture_weights /= np.sum(mixture_weights)
    mixture_weights = np.clip(mixture_weights, a_min = 1e-100, a_max = 1 - 1e-100)
    
    return (mixture_weights, multinom_probs)

# lambdas_init, theta_init = multinom_mix_em_init(10, 3)
# np.sum(lambdas_init) # should be 1

@njit
def posterior(log_lambda_cd: np.array, loglik: float) -> tuple[np.array, float]:
    k = log_lambda_cd.shape[0] # number of clusters
    n = log_lambda_cd.shape[1] # number of total observations in dataset
    post = np.empty((k,n)) # P(Z_j = 1 | X_i, Theta)
    
    for col_idx in range(n):
        col = log_lambda_cd[:, col_idx]
        max_j = np.argmax(col)
        curr_sum = 1.0
        
        for j in range(k):
            if j != max_j:
                val = np.exp(col[j] - col[max_j])
                curr_sum += val
                post[j, col_idx] = val
                
        loglik += np.log(curr_sum) + col[max_j]
        
        for j in range(k):
            if j == max_j:
                post[j, col_idx] = 1.0 / curr_sum
            else:
                post[j, col_idx] /= curr_sum
        
    return post, loglik
    

def multinom_mix_em(X: np.array,
                    k: int,
                    maxit=10e3,
                    epsilon=1e-03,
                    verb=True) -> tuple[np.array, np.array, list, np.array, bool]:
    """ Fit a Multinomial Mixture Model with k components using EM algorithm

    Args:
        X (np.array): nxp matrix of count observations.
        k (int): Number of components.
        maxit (int, optional): Maximum number of iterations for EM algorithm. Defaults to 10e3.
        epsilon (float, optional): Likelihood convergence threshold. Defaults to 1e-03.
        verb (bool, optional): Enabls verbose output with some additional information. Defaults to True.

    Returns:
        tuple[np.array, np.array, list, np.array, bool]:
        mixture weights, multinomial parameters, loglikehihoods, posterior component membership probability, and convergence flag
    """        
    n = X.shape[0] # n observations
    p = X.shape[1] # each multinomial has p categories (ie "predictors")
    col_sums = np.sum(X, axis=0) 
    row_sums = np.sum(X, axis=1) # num trials for each multinomial
    converged = True
    
    # initial parameter values
    #      [k]         [k, p]
    mixture_weights, multinom_probs = multinom_mix_em_init(k, p)
    
    # compute log-likelihood constant
    vectorized_lgamma = np.vectorize(math.lgamma)
    llconstant = np.sum(vectorized_lgamma(row_sums + 1)) - np.sum(vectorized_lgamma(X + 1))
    
    diff =  epsilon + 1 # init to enter EM loop
    new_loglik = 0
    iter = 0
    LL = []
    while (iter < maxit) and  (diff > epsilon):
        iter += 1
        old_loglik = new_loglik
        
        # E-Step: compute posterior P(z_j = 1 | x_i, theta)
        log_lambda_cd = np.log(mixture_weights)[:, np.newaxis] + np.log(multinom_probs) @ X.T  # [k, n] (colsum is loglikelihood without constant for one obs)
        
        # build posterior using the numerically stable algorithm
        post, new_loglik = posterior(log_lambda_cd, llconstant)
        
        # M-Step:
        multinom_probs = post @ X   # [k, n] x [n, p] = [k, p]
        multinom_probs /= np.sum(multinom_probs, axis=1)[:, np.newaxis]
        multinom_probs = np.clip(multinom_probs, a_min=1e-100, a_max=1-1e-100) # prevent 0's
        
        mixture_weights = np.mean(post, axis=1)
        
        
        # compute LL diff
        diff = np.abs(new_loglik - old_loglik)
        
        LL.append(new_loglik)
        
    if iter == maxit or np.isnan(new_loglik):
        print(f"Did not converge after {iter} iterations")
        converged = False
    else:
        if verb:
            print(f"Converged after {iter} iterations")
    
    if verb:
        print(f"loglikelihood: {new_loglik}")
    
    return mixture_weights, multinom_probs, LL, post, converged





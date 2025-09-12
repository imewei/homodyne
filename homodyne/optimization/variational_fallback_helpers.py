"""
Helper functions for NumPy fallback VI implementation
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple


def compute_numpy_elbo(variational_params: Dict[str, np.ndarray], 
                      data: np.ndarray, sigma: np.ndarray,
                      t1: np.ndarray, t2: np.ndarray, phi: np.ndarray,
                      q: float, L: float,
                      engine, param_priors, param_bounds, 
                      contrast_prior, contrast_bounds,
                      offset_prior, offset_bounds) -> Tuple[float, float, float]:
    """
    Compute ELBO using NumPy for variational inference fallback.
    
    Returns:
        (elbo, kl_divergence, likelihood)
    """
    # Extract variational parameters
    param_mu = variational_params['param_mu']
    param_log_std = variational_params['param_log_std']
    contrast_mu = variational_params['contrast_mu']
    contrast_log_std = variational_params['contrast_log_std']
    offset_mu = variational_params['offset_mu']
    offset_log_std = variational_params['offset_log_std']
    
    # Convert log std to std
    param_std = np.exp(param_log_std)
    contrast_std = np.exp(contrast_log_std)
    offset_std = np.exp(offset_log_std)
    
    # Sample from variational distributions (single sample for speed)
    param_sample = param_mu  # Use mean for deterministic estimate
    contrast_sample = contrast_mu
    offset_sample = offset_mu
    
    # Compute likelihood term
    try:
        g1_theory = engine.theory_engine.compute_g1(param_sample, t1, t2, phi, q, L)
        g2_theory = g1_theory**2 * contrast_sample + offset_sample
        
        residuals = (data - g2_theory) / sigma
        log_likelihood = -0.5 * np.sum(residuals**2) - 0.5 * np.sum(np.log(2 * np.pi * sigma**2))
    except Exception as e:
        # Penalize invalid parameters
        log_likelihood = -1e10
    
    # Compute KL divergence terms
    kl_divergence = 0.0
    
    # KL for physical parameters
    for i, (prior_mu, prior_std) in enumerate(param_priors):
        # Analytical KL for Gaussian distributions
        kl_i = (np.log(prior_std / param_std[i]) + 
                (param_std[i]**2 + (param_mu[i] - prior_mu)**2) / (2 * prior_std**2) - 0.5)
        kl_divergence += kl_i
    
    # KL for contrast
    prior_contrast_mu, prior_contrast_std = contrast_prior
    kl_contrast = (np.log(prior_contrast_std / contrast_std) +
                  (contrast_std**2 + (contrast_mu - prior_contrast_mu)**2) / (2 * prior_contrast_std**2) - 0.5)
    kl_divergence += kl_contrast
    
    # KL for offset
    prior_offset_mu, prior_offset_std = offset_prior
    kl_offset = (np.log(prior_offset_std / offset_std) +
                (offset_std**2 + (offset_mu - prior_offset_mu)**2) / (2 * prior_offset_std**2) - 0.5)
    kl_divergence += kl_offset
    
    # ELBO = likelihood - KL divergence
    elbo = log_likelihood - kl_divergence
    
    return elbo, kl_divergence, log_likelihood


def flatten_variational_params(params: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten variational parameters into single vector for optimization."""
    flat_params = []
    flat_params.extend(params['param_mu'].flatten())
    flat_params.extend(params['param_log_std'].flatten())
    flat_params.append(float(params['contrast_mu']))
    flat_params.append(float(params['contrast_log_std']))
    flat_params.append(float(params['offset_mu']))
    flat_params.append(float(params['offset_log_std']))
    return np.array(flat_params)


def unflatten_variational_params(flat_params: np.ndarray, n_params: int) -> Dict[str, np.ndarray]:
    """Unflatten parameter vector back to dictionary structure."""
    idx = 0
    
    param_mu = flat_params[idx:idx+n_params]
    idx += n_params
    
    param_log_std = flat_params[idx:idx+n_params]  
    idx += n_params
    
    contrast_mu = flat_params[idx]
    idx += 1
    
    contrast_log_std = flat_params[idx]
    idx += 1
    
    offset_mu = flat_params[idx]
    idx += 1
    
    offset_log_std = flat_params[idx]
    
    return {
        'param_mu': param_mu,
        'param_log_std': param_log_std,
        'contrast_mu': contrast_mu,
        'contrast_log_std': contrast_log_std,
        'offset_mu': offset_mu,
        'offset_log_std': offset_log_std,
    }
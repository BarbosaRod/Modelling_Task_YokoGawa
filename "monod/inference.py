"""
Bayesian hierarchical Parameter Estimation for Monod Kinetic Model

This module implements a hierarchical Bayesian inference approach to estimate 
parameters of the Monod kinetic model from experimental data.

The model uses:
- Shared parameters across experiments (μ_max, Ks)
- Process-specific yield coefficients (Yxs)
- Hierarchical prior structure to pool information across experiments
- NumPyro/JAX for efficient MCMC sampling
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def preprocess_experimental_data(processes):
    """
    Preprocess experimental data for Bayesian inference.
    
    Args:
        processes (list): List of dictionaries containing experimental data
            Each process should have:
            - time_points: array of time points
            - S: array of substrate measurements (may contain NaN)
            - X: array of metabolite measurements (may contain NaN)
            - feed_times: array of feeding time points
            - Vb: volume of feed
            - S_feed: substrate concentration in feed
            - y0: initial conditions [V0, S0, X0]
    
    Returns:
        list: Preprocessed process data with masks for valid observations
    """
    for proc in processes:
        # Create masks for valid (non-NaN) measurements
        valid_S_idx = ~jnp.isnan(proc["S"])
        valid_X_idx = ~jnp.isnan(proc["X"])
        
        # Store masked observations and masks
        proc["S_obs_masked"] = proc["S"][valid_S_idx]
        proc["X_obs_masked"] = proc["X"][valid_X_idx]
        proc["S_mask"] = valid_S_idx
        proc["X_mask"] = valid_X_idx
        
    return processes


def hierarchical_model(processes):
    """
    Hierarchical Bayesian model for Monod kinetic parameters.
    
    This model uses:
    1. Shared kinetic parameters (μ_max, Ks) across all processes
    2. Process-specific yield coefficients (Yxs)
    3. Hierarchical prior structure for Yxs
    
    Args:
        processes (list): List of preprocessed experimental datasets
    """
    # ---------- GLOBAL PARAMETERS ----------
    # Shared across all processes
    mu_max = numpyro.sample("mu_max", dist.LogNormal(-0.5, 0.5))
    Ks = numpyro.sample("Ks", dist.LogNormal(-2.0, 0.5))
    
    # ---------- HIERARCHICAL STRUCTURE FOR Yxs ----------
    # Population-level parameters for Yxs
    Yxs_mu = numpyro.sample("Yxs_mu", dist.LogNormal(0.0, 0.5))
    Yxs_sigma = numpyro.sample("Yxs_sigma", dist.HalfNormal(0.5))
    
    # ---------- PROCESS-SPECIFIC PARAMETERS AND LIKELIHOOD ----------
    for i, proc in enumerate(processes):
        # Sample process-specific Yxs from population distribution
        Yxs = numpyro.sample(f"Yxs_{i}", dist.LogNormal(Yxs_mu, Yxs_sigma))
        
        # Simulate the Monod model with current parameter samples
        S_pred, X_pred = simulate_monod(
            mu_max, Ks, Yxs,
            proc["time_points"], proc["feed_times"],
            proc["Vb"], proc["S_feed"], proc["y0"]
        )
        
        # Calculate likelihood for observed data points
        # Only compare model predictions with available measurements (masked)
        numpyro.sample(
            f"S_obs_{i}", 
            dist.Normal(S_pred[proc["S_mask"]], 0.1), 
            obs=proc["S_obs_masked"]
        ) #Sigma given
        
        numpyro.sample(
            f"X_obs_{i}", 
            dist.Normal(X_pred[proc["X_mask"]], 0.1), 
            obs=proc["X_obs_masked"]
        ) #Sigma given


def run_parameter_estimation(processes, num_warmup=500, num_samples=1000, 
                            num_chains=4, random_seed=0):
    """
    Run Bayesian parameter estimation using MCMC.
    
    Args:
        processes (list): List of experimental datasets
        num_warmup (int): Number of warmup iterations
        num_samples (int): Number of posterior samples to collect
        num_chains (int): Number of MCMC chains to run
        random_seed (int): Random seed for reproducibility
        
    Returns:
        dict: Posterior samples for model parameters
    """
    # Preprocess experimental data
    processed_data = preprocess_experimental_data(processes)
    
    # Initialize random number generator
    rng_key = jax.random.PRNGKey(random_seed)
    
    # Set up MCMC sampler
    kernel = NUTS(hierarchical_model)
    mcmc = MCMC(
        kernel, 
        num_warmup=num_warmup, 
        num_samples=num_samples,
        num_chains=num_chains
    )
    
    # Run MCMC
    mcmc.run(rng_key, processes=processed_data)
    
    # Get and return posterior samples
    samples = mcmc.get_samples()
    
    return samples


# Example usage
if __name__ == "__main__":
    # Example with minimal iterations for testing
    # In practice, use larger num_warmup and num_samples
    samples = run_parameter_estimation(
        processes, 
        num_warmup=5,   
        num_samples=5, 
        random_seed=0
    )
    
    # Print summary statistics for posterior samples
    for param, values in samples.items():
        print(f"{param}: mean = {values.mean():.4f}, std = {values.std():.4f}")

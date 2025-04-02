Monod Kinetic Model – Fed-Batch Bioprocess Simulation and Bayesian Inference
=============================================================================

This repository implements a complete modeling pipeline for fed-batch bioprocess simulation using Monod kinetics. It includes Bayesian parameter inference, uncertainty analysis, cross-validation, sensitivity analysis, and feed strategy optimization.

-------------------------------------------------------------------------------
Repository Structure
-------------------------------------------------------------------------------

monod/
├── KS_param_all.txt            # Posterior samples from Bayesian inference
├── cross_validation.py         # Leave-one-out cross-validation
├── fitting.py                  # MCMC setup and parameter estimation
├── inference.py                # Posterior loading, summaries, and visualization
├── model.py                    # Core Monod ODE model with feeding logic
├── optimisation                # Bayesian optimization of feed parameters
├── sensitivity_analysis        # Uncertainty and sensitivity analysis

-------------------------------------------------------------------------------
Overview
-------------------------------------------------------------------------------

The project is centered around a Monod-based ODE model for simulating substrate consumption S and metabolite production X in a fed-batch bioreactor.

It includes:
- Monod growth kinetics
- Fed-batch simulation using diffrax
- Parameter estimation using hierarchical Bayesian inference (NumPyro)
- Feed strategy optimization using Bayesian optimization (skopt)

-------------------------------------------------------------------------------
Model Description
-------------------------------------------------------------------------------

State variables:
- V   : Reactor volume (L)
- S   : Substrate concentration (mmol/L)
- X   : Metabolite (e.g. biomass or product) concentration (mmol/L)

Model parameters:
- μ_max : Maximum specific growth rate (hr⁻¹)
- Ks     : Substrate affinity constant (mmol/L)
- Yxs    : Biomass/substrate yield coefficient (mmol/mmol)

Feeding is implemented as bolus additions at discrete time points, with configurable feed volume (Vb) and substrate concentration (S_feed).

-------------------------------------------------------------------------------
Bayesian Inference
-------------------------------------------------------------------------------

Parameter estimation is performed using a hierarchical Bayesian model:

- Shared across processes: μ_max, Ks
- Process-specific: Yxs

Posterior samples from the MCMC run are stored in:
    KS_param_all.txt

These posteriors are used to:
- Quantify parameter uncertainty
- Visualize parameter distributions (inference.py)
- Perform leave-one-out cross-validation (cross_validation.py)
- Drive feed optimization (optimisation)
- Support robustness analysis (sensitivity_analysis)

-------------------------------------------------------------------------------
Simulation Example
-------------------------------------------------------------------------------

The model can be simulated using:

    from model import simulate_monod

    S, X = simulate_monod(
        mu_max=0.5,
        Ks=10.0,
        Yxs=0.8,
        times=...,
        feed_times=...,
        Vb=0.1,
        S_feed=100.0,
        y0=[1.0, 20.0, 1.0]
    )

-------------------------------------------------------------------------------
Usage Notes
-------------------------------------------------------------------------------

- All computations are implemented in JAX for performance and compatibility with automatic differentiation.
- The ODE solver is based on diffrax.Tsit5.
- Feeding logic is explicitly implemented and applied between integration steps.
- Posterior samples (KS_param_all.txt) are read and parsed using utilities in inference.py and cross_validation.py.

-------------------------------------------------------------------------------
Requirements
-------------------------------------------------------------------------------

- jax
- numpyro
- diffrax
- matplotlib
- seaborn
- skopt
- scikit-learn
- pandas

-------------------------------------------------------------------------------
License
-------------------------------------------------------------------------------

Proprietary – for internal or academic use only.

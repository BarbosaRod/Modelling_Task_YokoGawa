"""
Bayesian Optimization for Monod Kinetic Model

This module implements Bayesian optimization to find optimal feed parameters
(Feed concentration of S)
for the Monod kinetic model. It leverages the parameter estimates from the
Bayesian inference to optimize process conditions.

Key features:
- Feed volume (Vb) and substrate concentration (S_feed) optimization
- Multi-objective optimization capabilities
- Visualization of optimization results
- Uncertainty quantification in optimal parameters
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from skopt import gp_minimize, plots
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective
from model import *



def create_objective_function(proc, params, objective_type="final_X"):
    """
    Create an objective function for Bayesian optimization.
    
    Args:
        proc (dict): Process configuration dictionary
        params (dict): Model parameters (mu_max, Ks, Yxs)
        objective_type (str): Type of objective:
            - "final_X": Maximize final metabolite concentration
            - "max_X": Maximize maximum metabolite concentration
            - "yield": Maximize yield (X produced / S consumed)
            - "productivity": Maximize productivity (X produced / time)
            
    Returns:
        function: Objective function that takes Vb and S_feed as inputs
    """
    def objective(Vb, S_feed):
        # Make input array-like for consistency
        Vb_val = Vb if hasattr(Vb, "__len__") else float(Vb)
        S_feed_val = S_feed if hasattr(S_feed, "__len__") else float(S_feed)
        
        # Set up feed times from process configuration
        feed_times = proc["feed_times"]
        
        # Simulate the process with the given parameters
        S_sim, X_sim = simulate_monod(
            mu_max=params["mu_max"],
            Ks=params["Ks"],
            Yxs=params["Yxs"],
            times=proc["time_points"],
            feed_times=feed_times,
            Vb=Vb_val,
            S_feed=S_feed_val,
            y0=proc["y0"]
        )
        
        # Calculate the objective based on the selected type
        if objective_type == "final_X":
            # Maximize final metabolite concentration
            return X_sim[-1]
        
        elif objective_type == "max_X":
            # Maximize maximum metabolite concentration
            return jnp.max(X_sim)
        
        elif objective_type == "yield":
            # Maximize yield (X produced / S consumed)
            X_produced = X_sim[-1] - X_sim[0]
            S_consumed = S_sim[0] - S_sim[-1]
            # Avoid division by zero
            if S_consumed <= 0:
                return 0.0
            return X_produced / S_consumed
        
        elif objective_type == "productivity":
            # Maximize productivity (X produced / time)
            X_produced = X_sim[-1] - X_sim[0]
            total_time = proc["time_points"][-1] - proc["time_points"][0]
            return X_produced / total_time
        
        else:
            raise ValueError(f"Unknown objective type: {objective_type}")
    
    return objective


def optimize_feed_parameters(proc, params, Vb_range=(0.001, 0.03), S_feed_range=(5.0, 25.0),
                           objective_type="final_X", n_calls=30, random_state=42):
    """
    Perform Bayesian optimization to find optimal feed parameters.
    
    Args:
        proc (dict): Process configuration dictionary
        params (dict): Model parameters (mu_max, Ks, Yxs)
        Vb_range (tuple): Range for feed volume (min, max)
        S_feed_range (tuple): Range for substrate concentration (min, max)
        objective_type (str): Type of objective function
        n_calls (int): Number of optimization iterations
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (result, optimal_Vb, optimal_S_feed, optimal_value)
            - result: Full optimization result object
            - optimal_Vb: Optimal feed volume
            - optimal_S_feed: Optimal substrate concentration
            - optimal_value: Value of objective function at optimum
    """
    # Handle case where proc is a list of dictionaries (extract first process)
    if isinstance(proc, list):
        if len(proc) == 0:
            raise ValueError("Empty process list provided")
        proc = proc[0]  # Use the first process in the list
    
    # Define the search space
    space = [
        Real(Vb_range[0], Vb_range[1], name='Vb', prior='log-uniform'),
        Real(S_feed_range[0], S_feed_range[1], name='S_feed', prior='uniform')
    ]
    
    # Create objective function
    obj_func = create_objective_function(proc, params, objective_type)
    
    # Define the objective function for skopt (negative because we're maximizing)
    @use_named_args(space)
    def neg_objective(**kwargs):
        result = obj_func(kwargs["Vb"], kwargs["S_feed"])
        return -float(result)  # Minimize negative value → maximize value
    
    # Run Bayesian optimization
    result = gp_minimize(
        neg_objective, 
        space, 
        n_calls=n_calls,
        random_state=random_state,
        verbose=True
    )
    
    # Extract optimal parameters
    optimal_Vb = result.x[0]
    optimal_S_feed = result.x[1]
    optimal_value = -result.fun  # Convert back to positive value
    
    return result, optimal_Vb, optimal_S_feed, optimal_value


def plot_optimization_results(result, proc, params, optimal_Vb, optimal_S_feed, 
                             objective_type="final_X", show_simulations=True):
    """
    Plot the optimization results.
    
    Args:
        result: Optimization result from skopt
        proc (dict): Process configuration dictionary
        params (dict): Model parameters
        optimal_Vb (float): Optimal feed volume
        optimal_S_feed (float): Optimal substrate concentration
        objective_type (str): Type of objective function
        show_simulations (bool): Whether to show simulations with optimal parameters
        
    Returns:
        list: List of figure objects
    """
    # Handle case where proc is a list of dictionaries
    if isinstance(proc, list):
        if len(proc) == 0:
            raise ValueError("Empty process list provided")
        proc = proc[0]  # Use the first process in the list
    
    sns.set(style="whitegrid", context="notebook", font_scale=1.2)
    figures = []
    
    # Create figure for convergence plot
    fig_convergence = plt.figure(figsize=(10, 6))
    plot_convergence(result)
    plt.title(f"Convergence Plot ({objective_type})")
    plt.tight_layout()
    figures.append(fig_convergence)
    
    # Create figure for objective surface
    fig_objective = plt.figure(figsize=(15, 10))
    plot_objective(result, n_points=40)
    plt.suptitle(f"Objective Surface ({objective_type})", y=1.02, fontsize=16)
    plt.tight_layout()
    figures.append(fig_objective)
    
    if show_simulations:
        # Simulate the process with optimal parameters
        S_opt, X_opt = simulate_monod(
            mu_max=params["mu_max"],
            Ks=params["Ks"],
            Yxs=params["Yxs"],
            times=proc["time_points"],
            feed_times=proc["feed_times"],
            Vb=optimal_Vb,
            S_feed=optimal_S_feed,
            y0=proc["y0"]
        )
        
        # Simulate with original parameters for comparison
        S_orig, X_orig = simulate_monod(
            mu_max=params["mu_max"],
            Ks=params["Ks"],
            Yxs=params["Yxs"],
            times=proc["time_points"],
            feed_times=proc["feed_times"],
            Vb=proc.get("Vb", 0.01),  # Default if not provided
            S_feed=proc.get("S_feed", 10.0),  # Default if not provided
            y0=proc["y0"]
        )
        
        # Create figure for simulation comparison
        fig_sim = plt.figure(figsize=(12, 8))
        
        # Plot substrate
        plt.subplot(2, 1, 1)
        plt.plot(proc["time_points"], S_orig, 'b-', label='Original S')
        plt.plot(proc["time_points"], S_opt, 'g-', label='Optimized S')
        plt.ylabel('Substrate (mmol/L)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Mark feeding times
        for ft in proc["feed_times"]:
            plt.axvline(x=ft, color='gray', linestyle='--', alpha=0.5)
        
        # Plot metabolite
        plt.subplot(2, 1, 2)
        plt.plot(proc["time_points"], X_orig, 'b-', label='Original X')
        plt.plot(proc["time_points"], X_opt, 'g-', label='Optimized X')
        plt.xlabel('Time (hr)')
        plt.ylabel('Metabolite (mmol/L)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Mark feeding times
        for ft in proc["feed_times"]:
            plt.axvline(x=ft, color='gray', linestyle='--', alpha=0.5)
        
        # Add text with optimization results
        if objective_type == "final_X":
            improvement = (X_opt[-1] / X_orig[-1] - 1) * 100
            text = f"Optimal Vb: {optimal_Vb:.4f}\nOptimal S_feed: {optimal_S_feed:.2f}\n"
            text += f"Final X (orig): {X_orig[-1]:.4f}\nFinal X (opt): {X_opt[-1]:.4f}\n"
            text += f"Improvement: {improvement:.1f}%"
        else:
            text = f"Optimal Vb: {optimal_Vb:.4f}\nOptimal S_feed: {optimal_S_feed:.2f}"
        
        plt.figtext(0.02, 0.02, text, fontsize=12,
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.suptitle(f"Process Simulation with Optimal Parameters ({proc['name']})", 
                    y=0.98, fontsize=16)
        plt.tight_layout()
        figures.append(fig_sim)
    
    return figures


def parameter_uncertainty_analysis(proc, parameter_samples, n_samples=20, 
                                  Vb_range=(0.001, 0.03), S_feed_range=(5.0, 25.0),
                                  objective_type="final_X", n_calls=20,
                                  random_state=42):
    """
    Perform uncertainty analysis by optimizing with different parameter samples.
    
    Args:
        proc (dict): Process configuration dictionary or list of process dictionaries
        parameter_samples (dict): Dictionary of parameter samples from MCMC
        n_samples (int): Number of parameter sets to sample
        Vb_range (tuple): Range for feed volume (min, max)
        S_feed_range (tuple): Range for substrate concentration (min, max)
        objective_type (str): Type of objective function
        n_calls (int): Number of optimization iterations per sample
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (results_df, figures)
            - results_df: DataFrame with optimization results for each sample
            - figures: List of figure objects
    """
    import pandas as pd
    
    # Handle case where proc is a list of dictionaries
    if isinstance(proc, list):
        if len(proc) == 0:
            raise ValueError("Empty process list provided")
        proc = proc[0]  # Use the first process in the list
    
    # Set random seed
    np.random.seed(random_state)
    
    # Get random indices for parameter samples
    idx = np.random.choice(len(parameter_samples["mu_max"]), size=n_samples, replace=False)
    
    # Prepare results storage
    results = []
    optimal_Vbs = []
    optimal_S_feeds = []
    optimal_values = []
    
    # Process name for Ks parameter selection
    proc_name = proc["name"]
    proc_num = proc_name.split("_")[-1]
    Ks_key = f"Ks_process_{proc_num}"
    
    # Run optimization with different parameter samples
    for i, sample_idx in enumerate(idx):
        print(f"\nRunning optimization with parameter sample {i+1}/{n_samples}")
        
        # Extract parameters for this sample
        mu_max_i = float(parameter_samples["mu_max"][sample_idx])
        
        # Get Ks from process-specific samples if available, otherwise use global Ks
        if Ks_key in parameter_samples:
            Ks_i = float(parameter_samples[Ks_key][sample_idx])
        else:
            Ks_i = float(parameter_samples.get("Ks_mu", parameter_samples.get("Ks"))[sample_idx])
        
        # Get Yxs (either process-specific or global)
        Yxs_key = f"Yxs_{proc_num}"
        if Yxs_key in parameter_samples:
            Yxs_i = float(parameter_samples[Yxs_key][sample_idx])
        else:
            Yxs_i = float(parameter_samples["Yxs"][sample_idx])
        
        # Create parameter dictionary
        params_i = {
            "mu_max": mu_max_i,
            "Ks": Ks_i,
            "Yxs": Yxs_i
        }
        
        # Run optimization for this parameter set
        try:
            _, opt_Vb, opt_S_feed, opt_value = optimize_feed_parameters(
                proc, params_i, 
                Vb_range=Vb_range, 
                S_feed_range=S_feed_range,
                objective_type=objective_type, 
                n_calls=n_calls,
                random_state=random_state + i  # Different seed for each run
            )
            
            # Store results
            results.append({
                "sample_idx": sample_idx,
                "mu_max": mu_max_i,
                "Ks": Ks_i,
                "Yxs": Yxs_i,
                "optimal_Vb": opt_Vb,
                "optimal_S_feed": opt_S_feed,
                "optimal_value": opt_value
            })
            
            optimal_Vbs.append(opt_Vb)
            optimal_S_feeds.append(opt_S_feed)
            optimal_values.append(opt_value)
            
        except Exception as e:
            print(f"Optimization failed for sample {i+1}: {e}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create visualization
    figures = []
    
    # Plot distributions of optimal parameters
    fig_dist, axes = plt.subplots(1, 3, figsize=(15, 15))
    
    # Optimal Vb distribution
    sns.histplot(optimal_Vbs, kde=True, ax=axes[0])
    axes[0].set_title("Distribution of Optimal Feed Volume (Vb)")
    axes[0].set_xlabel("Vb")
    
    # Optimal S_feed distribution
    sns.histplot(optimal_S_feeds, kde=True, ax=axes[1])
    axes[1].set_title("Distribution of Optimal Feed Concentration (S_feed)")
    axes[1].set_xlabel("S_feed (mmol/L)")
    
    # Optimal objective value distribution
    sns.histplot(optimal_values, kde=True, ax=axes[2])
    axes[2].set_title(f"Distribution of Optimal {objective_type}")
    axes[2].set_xlabel("Value")
    
    plt.tight_layout()
    figures.append(fig_dist)
    
    # Plot scatter of optimal parameters
    fig_scatter = plt.figure(figsize=(8, 6))
    scatter = plt.scatter(optimal_Vbs, optimal_S_feeds, c=optimal_values, 
                         cmap='viridis', alpha=0.8, s=80)
    cbar = plt.colorbar(scatter)
    cbar.set_label(f"Optimal {objective_type}")
    plt.xlabel("Optimal Feed Volume (Vb)")
    plt.ylabel("Optimal Feed Concentration (S_feed)")
    plt.title("Optimal Parameters Across Posterior Samples")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    figures.append(fig_scatter)
    
    return results_df, figures


# Example usage
if __name__ == "__main__":
    import re
    
    # Load parameters from file
    def load_parameter_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        clean_content = re.sub(r'Array\((\[[\s\S]*?\]),\s*dtype=float32\)', r'np.array(\1)', content)
        data = eval(clean_content, {"np": np})
        return data
    
    # Load parameters
    param_file = 'KS_param_all.txt'
    data = load_parameter_file(param_file)
    
    # Extract median parameters
    Ks_process_5_median = float(np.median(data['Ks_process_5']))
    mu_max_median = float(np.median(data['mu_max']))
    Yxs_median = float(np.median(data['Yxs']))
    
    # Set up parameters
    median_params = {
        "mu_max": mu_max_median,
        "Ks": Ks_process_5_median,
        "Yxs": Yxs_median,
    }
    
    # Example process (as a list, similar to your error case)
    proc = [
                  {
        "name": "process_5",
        "time_points": jnp.array([0, 12, 12.01, 24, 24.01, 36, 36.01, 48, 48.01,
                                  60, 60.01, 72, 72.01, 84, 84.01, 96, 96.01, 108, 108.01]),
        "S": jnp.array([1.0222119, 1.0092245, 1.278277875, 1.1992364, 1.459628166,
                        1.1060543, 1.363349591, 1.1117914, 1.364304284,
                        0.82188684, 1.075067432, 0.391334, 0.647626386,
                        0.09087138, 0.347925322, 0.1124436, 0.364775064,
                        0.055019245, 0.304102258]),
        "X": jnp.array([0.08, 0.29, 0.29, 0.17, 0.17, 0.54, 0.53, 0.97, 0.96,
                        1.49, 1.47, 2.30, 2.26, 2.74, 2.69, 3.19, 3.14, 3.42, 3.36]),
        "feed_times": jnp.array([12, 24, 36, 48, 60, 72, 84, 96, 108]),
        "Vb": 0.02,
        "S_feed": 15.0,
        "y0": jnp.array([1, 1.0222119, 0.08])}
    ]
    
    # Run optimization - will work with both a single process dict or a list of processes
    result, optimal_Vb, optimal_S_feed, optimal_value = optimize_feed_parameters(
        proc, median_params, n_calls=20)
    
    print(f"Optimal Vb: {optimal_Vb:.4f}")
    print(f"Optimal S_feed: {optimal_S_feed:.2f}")
    print(f"Optimal value: {optimal_value:.4f}")
    
    # Plot results
    figures = plot_optimization_results(
        result, proc, median_params, optimal_Vb, optimal_S_feed)

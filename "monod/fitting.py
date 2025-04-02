"""
Bayesian Model Utilities

This module provides utility functions for:
1. Loading and parsing parameter files from previous MCMC runs
2. Visualizing Bayesian model fits and parameter distributions
3. Calculating metrics for model evaluation
"""

import re
import numpy as np
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from model import *
from processes import processes


def load_parameter_file(file_path):
    """
    Load and parse a parameter file containing MCMC samples.
    
    Args:
        file_path (str): Path to the parameter file
        
    Returns:
        dict: Dictionary of parameter samples
    """
    # Read file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace JAX Array format with numpy arrays for safer eval
    clean_content = re.sub(r'Array\((\[[\s\S]*?\]),\s*dtype=float32\)', r'np.array(\1)', content)
    
    # Evaluate safely with restricted namespace
    data = eval(clean_content, {"np": np})
    
    return data


def compute_parameter_statistics(data):
    """
    Compute median values for parameters from MCMC samples.
    
    Args:
        data (dict): Dictionary of parameter samples
        
    Returns:
        dict: Dictionary of median parameter values
    """
    # Process-specific Ks parameters
    Ks_medians = {}
    for i in range(1, 6):  # Assuming 5 processes
        key = f'Ks_process_{i}'
        if key in data:
            Ks_medians[f"process_{i}"] = float(np.median(data[key]))
    
    # Global parameters
    medians = {
        "Ks_mu": float(np.median(data.get('Ks_mu', []))),
        "Ks_sigma": float(np.median(data.get('Ks_sigma', []))),
        "mu_max": float(np.median(data.get('mu_max', []))),
        "Yxs": float(np.median(data.get('Yxs', [])))
    }
    
    # Compute total Ks median across all processes
    all_Ks = []
    for i in range(1, 6):
        key = f'Ks_process_{i}'
        if key in data:
            all_Ks.append(data[key])
    
    if all_Ks:
        all_Ks_array = np.concatenate(all_Ks)
        medians["Ks_total"] = float(np.median(all_Ks_array))
    
    # Add process-specific medians to the output dictionary
    medians["Ks_process"] = Ks_medians
    
    return medians


def plot_parameter_distributions(data, medians=None, save_path=None):
    """
    Plot the distributions of parameters from MCMC samples.
    
    Args:
        data (dict): Dictionary of parameter samples
        medians (dict, optional): Pre-computed median values
        save_path (str, optional): Path to save the plot
    """
    # Set Seaborn style
    sns.set(style="white", context="notebook", font_scale=1.2)
    
    # Determine number of parameters to plot
    params_to_plot = ['mu_max', 'Ks_mu', 'Ks_sigma', 'Yxs']
    process_params = [key for key in data.keys() if key.startswith('Ks_process_')]
    
    n_plots = len(params_to_plot) + len(process_params)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
    if n_plots == 1:  # Handle case with only one plot
        axes = np.array([axes])
    axes = axes.flatten()
    
    # If medians not provided, compute them
    if medians is None:
        medians = compute_parameter_statistics(data)
    
    # Plot global parameters
    for i, param in enumerate(params_to_plot):
        if param in data:
            ax = axes[i]
            sns.histplot(data[param], kde=True, ax=ax)
            if param in medians:
                ax.axvline(medians[param], color='red', linestyle='--', 
                           label=f'Median: {medians[param]:.4f}')
            ax.set_title(f'Distribution of {param}')
            ax.legend()
    
    # Plot process-specific parameters
    for i, param in enumerate(process_params):
        ax = axes[i + len(params_to_plot)]
        sns.histplot(data[param], kde=True, ax=ax)
        proc_name = param.replace('Ks_', '')
        if proc_name in medians.get("Ks_process", {}):
            ax.axvline(medians["Ks_process"][proc_name], color='red', linestyle='--',
                       label=f'Median: {medians["Ks_process"][proc_name]:.4f}')
        ax.set_title(f'Distribution of {param}')
        ax.legend()
    
    # Remove any unused subplots
    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_model_fits(processes, data, medians=None, n_samples=100, seed=0, save_dir=None):
    """
    Plot model fits with prediction intervals.
    
    Args:
        processes (list): List of process dictionaries with experimental data
        data (dict): Dictionary of parameter samples
        medians (dict, optional): Pre-computed median values
        n_samples (int): Number of posterior samples to use for prediction intervals
        seed (int): Random seed for reproducibility
        save_dir (str, optional): Directory to save plots
    """
    # Set Seaborn style
    sns.set(style="white", context="notebook", font_scale=1.2)
    
    # If medians not provided, compute them
    if medians is None:
        medians = compute_parameter_statistics(data)
    
    # Generate random indices for sampling
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(data["mu_max"]), size=min(n_samples, len(data["mu_max"])), replace=False)
    
    # Plot each process
    for proc in processes:
        proc_name = proc["name"]
        print(f"Plotting {proc_name}...")
        
        # Get process-specific Ks median
        Ks_i_proc_median = medians["Ks_process"].get(proc_name, medians["Ks_total"])
        
        S_preds = []
        X_preds = []
        
        # Sample from posterior
        for i in idx:
            mu_max_i = data["mu_max"][i]
            Ks_i = Ks_i_proc_median  # Using process-specific Ks
            
            # If process-specific Yxs exists, use it
            Yxs_key = f"Yxs_{proc_name.split('_')[-1]}"
            Yxs_i = data[Yxs_key][i] if Yxs_key in data else data["Yxs"][i]
            
            S_i, X_i = simulate_monod(
                mu_max=mu_max_i,
                Ks=Ks_i,
                Yxs=Yxs_i,
                times=proc["time_points"],
                feed_times=proc["feed_times"],
                Vb=proc["Vb"],
                S_feed=proc["S_feed"],
                y0=proc["y0"]
            )
            S_preds.append(S_i)
            X_preds.append(X_i)
        
        S_preds = jnp.stack(S_preds)
        X_preds = jnp.stack(X_preds)
        
        # Add observational noise
        key = random.PRNGKey(seed)
        noise_scale = 0.1  # Observation noise scale
        key, subkey = random.split(key)
        noise_X = random.normal(subkey, X_preds.shape) * noise_scale
        key, subkey = random.split(key)
        noise_S = random.normal(subkey, S_preds.shape) * noise_scale
        
        X_pred_with_noise = X_preds + noise_X
        S_pred_with_noise = S_preds + noise_S
        
        # Compute prediction intervals
        percentiles = jnp.array([2.5, 50, 97.5])
        X_lower, X_median, X_upper = jnp.percentile(X_pred_with_noise, percentiles, axis=0)
        S_lower, S_median, S_upper = jnp.percentile(S_pred_with_noise, percentiles, axis=0)
        
        # Mask NaNs for metrics calculation
        mask_X = ~jnp.isnan(proc["X"])
        mask_S = ~jnp.isnan(proc["S"])
        
        # Calculate metrics
        rmse_X = np.sqrt(mean_squared_error(proc["X"][mask_X], X_median[mask_X]))
        rmse_S = np.sqrt(mean_squared_error(proc["S"][mask_S], S_median[mask_S]))
        
        r2_X = r2_score(proc["X"][mask_X], X_median[mask_X])
        r2_S = r2_score(proc["S"][mask_S], S_median[mask_S])
        
        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        
        # --- Substrate (S) ---
        ax = axes[0]
        ax.plot(proc["time_points"], proc["S"], 'o', label='Observed S', color='black')
        ax.plot(proc["time_points"], S_median, linestyle='--', label='Predicted S (median)', color='seagreen')
        ax.fill_between(proc["time_points"], S_lower, S_upper, alpha=0.2, color='seagreen', label='95% PI')
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Time (hr)")
        ax.set_ylabel("Substrate (mmol/L)")
        ax.text(0.05, 0.95, f"RMSE: {rmse_S:.2f}\nR²: {r2_S:.2f}",
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9))
        ax.legend(frameon=False)
        
        # --- Metabolite (X) ---
        ax = axes[1]
        ax.plot(proc["time_points"], proc["X"], 'o', label='Observed X', color='black')
        ax.plot(proc["time_points"], X_median, linestyle='--', label='Predicted X (median)', color='royalblue')
        ax.fill_between(proc["time_points"], X_lower, X_upper, alpha=0.2, color='royalblue', label='95% PI')
        ax.set_xlabel("Time (hr)")
        ax.set_ylabel("Metabolite (mmol/L)")
        ax.set_ylim(bottom=0)
        ax.text(0.05, 0.95, f"RMSE: {rmse_X:.2f}\nR²: {r2_X:.2f}",
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9))
        ax.legend(frameon=False)
        
        # Add feeding times if available
        for ft in proc.get("feed_times", []):
            axes[0].axvline(x=ft, color='green', linestyle='--', alpha=0.5)
            axes[1].axvline(x=ft, color='green', linestyle='--', alpha=0.5)
        
        plt.suptitle(f"Process: {proc_name}", y=1.03, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        
        if save_dir:
            plt.savefig(f"{save_dir}/fit_{proc_name}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        

def plot_process_specific_parameters(data, param_name="Ks", save_path=None):
    """
    Plot comparison of process-specific parameters.
    
    Args:
        data (dict): Dictionary of parameter samples
        param_name (str): Parameter name to compare across processes
        save_path (str, optional): Path to save the plot
    """
    sns.set(style="white", context="talk")
    
    # Collect process-specific parameters
    process_data = {}
    for key in data.keys():
        if key.startswith(f"{param_name}_process_"):
            process_num = key.split("_")[-1]
            process_data[f"Process {process_num}"] = data[key]
    
    if not process_data:
        print(f"No process-specific {param_name} parameters found.")
        return None
    
    # Create dataframe for plotting
    import pandas as pd
    df_list = []
    
    for process_name, values in process_data.items():
        df_list.append(pd.DataFrame({
            'Process': process_name,
            'Value': values
        }))
    
    df = pd.concat(df_list)
    
    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='Process', y='Value', data=df)
    ax.set_title(f"Comparison of {param_name} across processes")
    ax.set_ylabel(f"{param_name} value")
    
    # Add jittered points
    sns.stripplot(x='Process', y='Value', data=df, size=4, color=".3", alpha=0.4)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


# Example usage
if __name__ == "__main__":
    # Example code for using the utilities
    param_file = 'KS_param_all.txt'
    data = load_parameter_file(param_file)
    medians = compute_parameter_statistics(data)
    
    # Plot parameter distributions
    plot_parameter_distributions(data, medians, save_path="parameter_distributions.png")
    
    # Plot process-specific parameters
    plot_process_specific_parameters(data, param_name="Ks", save_path="ks_by_process.png")
    
    # Example processes definition
    processes = [
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
        "y0": jnp.array([1, 1.0222119, 0.08])
    }
    ]
    
    # Plot model fits
    plot_model_fits(processes, data, medians, save_dir="./plots")

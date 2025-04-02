"""
Sensitivity Analysis for Monod Kinetic Model

This module implements various sensitivity analysis methods for the Monod kinetic model:
1. Sobol sensitivity analysis (global, variance-based)
2. One-at-a-time (OAT) analysis
3. Morris method (elementary effects)

These methods help identify which parameters have the most significant impact on model outputs.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from SALib.sample import saltelli, morris
from SALib.analyze import sobol, morris as morris_analyze
from tqdm.auto import tqdm
from model import *
from processes import processes


def run_sobol_analysis(proc, baseline_params, param_ranges=None, N=512, 
                     output_type="final_X", calc_second_order=True):
    """
    Perform Sobol sensitivity analysis on the Monod kinetic model.
    
    Args:
        proc (dict): Process configuration dictionary
        baseline_params (dict): Baseline parameter values (mu_max, Ks, Yxs)
        param_ranges (dict, optional): Dictionary specifying parameter ranges as fractions
                                      of baseline (e.g., {'mu_max': (0.8, 1.2)})
        N (int): Sample size for Saltelli sampling
        output_type (str): Output metric to analyze ('final_X', 'max_X', 'yield', etc.)
        calc_second_order (bool): Whether to calculate second-order indices
    
    Returns:
        tuple: (Si, Y_output, param_values)
            - Si: Sobol sensitivity indices
            - Y_output: Model outputs for each parameter set
            - param_values: Parameter sets used for analysis
    """
    # Handle case where proc is a list of dictionaries
    if isinstance(proc, list):
        if len(proc) == 0:
            raise ValueError("Empty process list provided")
        proc = proc[0]  # Use the first process in the list
    
    # Extract baseline parameters
    mu_max_baseline = baseline_params["mu_max"]
    Ks_baseline = baseline_params["Ks"]
    Yxs_baseline = baseline_params["Yxs"]
    
    # Set default parameter ranges if not provided (±10% by default)
    if param_ranges is None:
        param_ranges = {
            'mu_max': (0.9, 1.1),  # ±10%
            'Ks': (0.9, 1.1),      # ±10%
            'Yxs': (0.9, 1.1)      # ±10%
        }
    
    # Define problem for SALib
    problem = {
        'num_vars': 3,
        'names': ['mu_max', 'Ks', 'Yxs'],
        'bounds': [
            [mu_max_baseline * param_ranges['mu_max'][0], 
             mu_max_baseline * param_ranges['mu_max'][1]],
            [Ks_baseline * param_ranges['Ks'][0], 
             Ks_baseline * param_ranges['Ks'][1]],
            [Yxs_baseline * param_ranges['Yxs'][0], 
             Yxs_baseline * param_ranges['Yxs'][1]]
        ]
    }
    
    # Generate parameter samples using Saltelli's extension of Sobol sequence
    param_values = saltelli.sample(problem, N=N, calc_second_order=calc_second_order)
    
    # Process configuration
    times = proc["time_points"]
    y0 = proc["y0"]
    feed_times = proc["feed_times"]
    Vb = proc["Vb"]
    S_feed = proc["S_feed"]
    
    # Run model for each parameter set
    Y_output = []
    
    for params in tqdm(param_values, desc="Running simulations"):
        mu_max, Ks, Yxs = params
        
        try:
            S, X = simulate_monod(mu_max, Ks, Yxs, times, feed_times, Vb, S_feed, y0)
            
            # Calculate output metric based on selected type
            if output_type == "final_X":
                Y_output.append(float(X[-1]))  # Final metabolite concentration
            elif output_type == "max_X":
                Y_output.append(float(X.max()))  # Maximum metabolite concentration
            elif output_type == "final_S":
                Y_output.append(float(S[-1]))  # Final substrate concentration
            elif output_type == "yield":
                # Yield = X produced / S consumed
                X_produced = float(X[-1] - X[0])
                S_consumed = float(S[0] - S[-1])
                if S_consumed <= 0:
                    Y_output.append(0.0)
                else:
                    Y_output.append(X_produced / S_consumed)
            else:
                # Default to final metabolite
                Y_output.append(float(X[-1]))
                
        except Exception as e:
            print(f"Simulation failed: {e}")
            Y_output.append(np.nan)
    
    # Convert to numpy array
    Y_output = np.array(Y_output)
    
    # Remove NaN values
    valid_indices = ~np.isnan(Y_output)
    Y_valid = Y_output[valid_indices]
    params_valid = param_values[valid_indices]
    
    if len(Y_valid) < len(Y_output):
        print(f"Warning: {len(Y_output) - len(Y_valid)} simulations failed and were removed")
    
    if len(Y_valid) < N:
        print(f"Warning: Only {len(Y_valid)} valid samples out of {N}")
    
    # Perform Sobol analysis
    Si = sobol.analyze(problem, Y_valid, calc_second_order=calc_second_order, 
                     print_to_console=False)
    
    return Si, Y_output, param_values


def plot_sobol_indices(Si, problem, title=None, save_path=None):
    """
    Plot Sobol sensitivity indices.
    
    Args:
        Si: Sobol indices from SALib
        problem: Problem definition
        title (str, optional): Plot title
        save_path (str, optional): Path to save the figure
        
    Returns:
        tuple: (fig, axes) Matplotlib figure and axes
    """
    sns.set(style="whitegrid", context="notebook", font_scale=1.2)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # First-order indices (S1)
    axes[0].bar(problem['names'], Si['S1'], yerr=Si['S1_conf'], capsize=4)
    axes[0].set_title("First-order Sobol Indices (S1)")
    axes[0].set_ylabel("Sensitivity Index")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    
    # Total-order indices (ST)
    axes[1].bar(problem['names'], Si['ST'], yerr=Si['ST_conf'], capsize=4)
    axes[1].set_title("Total-order Sobol Indices (ST)")
    axes[1].set_ylabel("Sensitivity Index")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=16, y=1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig, axes


def plot_sobol_second_order(Si, problem, title=None, save_path=None):
    """
    Plot second-order Sobol indices as a heatmap.
    
    Args:
        Si: Sobol indices from SALib
        problem: Problem definition
        title (str, optional): Plot title
        save_path (str, optional): Path to save the figure
        
    Returns:
        tuple: (fig, ax) Matplotlib figure and axis
    """
    sns.set(style="white", context="notebook", font_scale=1.2)
    
    if 'S2' not in Si:
        print("Second-order indices not available. Use calc_second_order=True in Sobol analysis.")
        return None, None
    
    # Get parameter names
    names = problem['names']
    n_params = len(names)
    
    # Create matrix for heatmap
    S2_matrix = np.zeros((n_params, n_params))
    
    # Fill the matrix with second-order indices
    indices = []
    for i in range(n_params):
        for j in range(i+1, n_params):
            idx = len(indices)
            if idx < len(Si['S2']):
                S2_matrix[i, j] = Si['S2'][idx]
                S2_matrix[j, i] = Si['S2'][idx]  # Symmetric
                indices.append((i, j))
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(S2_matrix, dtype=bool), k=1)  # Mask upper triangle
    heatmap = sns.heatmap(S2_matrix, annot=True, fmt=".3f", cmap="YlGnBu",
                        xticklabels=names, yticklabels=names, mask=mask,
                        vmin=0, vmax=S2_matrix.max() if S2_matrix.max() > 0 else 1,
                        cbar_kws={'label': 'Second-order Index'})
    
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig, ax


def run_oat_analysis(proc, baseline_params, param_ranges=None, n_points=21,
                   output_type="final_X"):
    """
    Perform One-At-a-Time (OAT) sensitivity analysis.
    
    Args:
        proc (dict): Process configuration dictionary
        baseline_params (dict): Baseline parameter values (mu_max, Ks, Yxs)
        param_ranges (dict, optional): Dictionary specifying parameter ranges as fractions
                                      of baseline (e.g., {'mu_max': (0.8, 1.2)})
        n_points (int): Number of points to sample for each parameter
        output_type (str): Output metric to analyze
        
    Returns:
        dict: Dictionary with OAT results for each parameter
    """
    # Handle case where proc is a list of dictionaries
    if isinstance(proc, list):
        if len(proc) == 0:
            raise ValueError("Empty process list provided")
        proc = proc[0]  # Use the first process in the list
    
    # Extract baseline parameters
    mu_max_baseline = baseline_params["mu_max"]
    Ks_baseline = baseline_params["Ks"]
    Yxs_baseline = baseline_params["Yxs"]
    
    # Set default parameter ranges if not provided (±30% by default for wider exploration)
    if param_ranges is None:
        param_ranges = {
            'mu_max': (0.7, 1.3),  # ±30%
            'Ks': (0.7, 1.3),      # ±30%
            'Yxs': (0.7, 1.3)      # ±30%
        }
    
    # Create parameter ranges for each parameter
    params_to_vary = {
        'mu_max': np.linspace(
            mu_max_baseline * param_ranges['mu_max'][0],
            mu_max_baseline * param_ranges['mu_max'][1],
            n_points
        ),
        'Ks': np.linspace(
            Ks_baseline * param_ranges['Ks'][0],
            Ks_baseline * param_ranges['Ks'][1],
            n_points
        ),
        'Yxs': np.linspace(
            Yxs_baseline * param_ranges['Yxs'][0],
            Yxs_baseline * param_ranges['Yxs'][1],
            n_points
        )
    }
    
    # Process configuration
    times = proc["time_points"]
    y0 = proc["y0"]
    feed_times = proc["feed_times"]
    Vb = proc["Vb"]
    S_feed = proc["S_feed"]
    
    # Run baseline simulation
    S_baseline, X_baseline = simulate_monod(
        mu_max_baseline, Ks_baseline, Yxs_baseline, 
        times, feed_times, Vb, S_feed, y0
    )
    
    # Calculate baseline output
    if output_type == "final_X":
        baseline_output = float(X_baseline[-1])
    elif output_type == "max_X":
        baseline_output = float(X_baseline.max())
    elif output_type == "final_S":
        baseline_output = float(S_baseline[-1])
    elif output_type == "yield":
        X_produced = float(X_baseline[-1] - X_baseline[0])
        S_consumed = float(S_baseline[0] - S_baseline[-1])
        if S_consumed <= 0:
            baseline_output = 0.0
        else:
            baseline_output = X_produced / S_consumed
    else:
        baseline_output = float(X_baseline[-1])
    
    # Dictionary to store results
    oat_results = {
        'mu_max': {'values': [], 'outputs': [], 'percent_change': []},
        'Ks': {'values': [], 'outputs': [], 'percent_change': []},
        'Yxs': {'values': [], 'outputs': [], 'percent_change': []}
    }
    
    # Run simulations varying each parameter one at a time
    for param_name, param_values in params_to_vary.items():
        for param_value in tqdm(param_values, desc=f"Varying {param_name}"):
            # Set parameters
            mu_max = param_value if param_name == 'mu_max' else mu_max_baseline
            Ks = param_value if param_name == 'Ks' else Ks_baseline
            Yxs = param_value if param_name == 'Yxs' else Yxs_baseline
            
            try:
                # Run simulation
                S, X = simulate_monod(mu_max, Ks, Yxs, times, feed_times, Vb, S_feed, y0)
                
                # Calculate output
                if output_type == "final_X":
                    output = float(X[-1])
                elif output_type == "max_X":
                    output = float(X.max())
                elif output_type == "final_S":
                    output = float(S[-1])
                elif output_type == "yield":
                    X_produced = float(X[-1] - X[0])
                    S_consumed = float(S[0] - S[-1])
                    if S_consumed <= 0:
                        output = 0.0
                    else:
                        output = X_produced / S_consumed
                else:
                    output = float(X[-1])
                
                # Calculate percent change from baseline
                percent_change = ((output - baseline_output) / baseline_output) * 100
                
                # Store results
                oat_results[param_name]['values'].append(param_value)
                oat_results[param_name]['outputs'].append(output)
                oat_results[param_name]['percent_change'].append(percent_change)
                
            except Exception as e:
                print(f"Simulation failed for {param_name}={param_value}: {e}")
    
    # Calculate normalized sensitivity coefficients
    for param_name in oat_results.keys():
        # Get baseline parameter value
        baseline_value = baseline_params[param_name]
        
        # Calculate normalized sensitivity coefficients
        values = np.array(oat_results[param_name]['values'])
        outputs = np.array(oat_results[param_name]['outputs'])
        
        # Skip if not enough valid points
        if len(values) < 2:
            oat_results[param_name]['sensitivity'] = None
            continue
        
        # Calculate normalized sensitivity
        param_rel_change = (values - baseline_value) / baseline_value
        output_rel_change = (outputs - baseline_output) / baseline_output
        
        # Use central difference for better accuracy (skip endpoints)
        valid_idx = np.where(param_rel_change != 0)[0]
        sensitivities = output_rel_change[valid_idx] / param_rel_change[valid_idx]
        
        # Store median sensitivity (more robust than mean)
        oat_results[param_name]['sensitivity'] = float(np.median(sensitivities))
    
    return oat_results, baseline_output


def plot_oat_results(oat_results, baseline_params, output_type="final_X", title=None, save_path=None):
    """
    Plot results from One-At-a-Time sensitivity analysis.
    
    Args:
        oat_results (dict): Results from OAT analysis
        baseline_params (dict): Baseline parameter values
        output_type (str): Type of output analyzed
        title (str, optional): Plot title
        save_path (str, optional): Path to save the figure
        
    Returns:
        tuple: (fig, axes) Matplotlib figure and axes
    """
    sns.set(style="whitegrid", context="notebook", font_scale=1.2)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot each parameter's effect
    for i, (param_name, results) in enumerate(oat_results.items()):
        ax = axes[i]
        
        # Skip if no valid results
        if not results['values']:
            continue
        
        # Extract data
        values = np.array(results['values'])
        percent_change = np.array(results['percent_change'])
        
        # Normalize x-axis to show percentage change from baseline
        baseline_value = baseline_params[param_name]
        x_values = ((values - baseline_value) / baseline_value) * 100
        
        # Plot
        ax.plot(x_values, percent_change, 'o-', linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        # Add text with sensitivity
        if results.get('sensitivity') is not None:
            sens_text = f"Sensitivity: {results['sensitivity']:.3f}"
            ax.text(0.05, 0.95, sens_text, transform=ax.transAxes, 
                  fontsize=12, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Labels
        ax.set_xlabel(f"% Change in {param_name}")
        ax.set_ylabel(f"% Change in {output_type}")
        ax.set_title(f"Effect of {param_name}")
        ax.grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=16, y=1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig, axes


def plot_oat_sensitivity_comparison(oat_results, output_type="final_X", title=None, save_path=None):
    """
    Plot a bar chart comparing sensitivities from OAT analysis.
    
    Args:
        oat_results (dict): Results from OAT analysis
        output_type (str): Type of output analyzed
        title (str, optional): Plot title
        save_path (str, optional): Path to save the figure
        
    Returns:
        tuple: (fig, ax) Matplotlib figure and axis
    """
    sns.set(style="whitegrid", context="notebook", font_scale=1.2)
    
    # Extract sensitivities
    param_names = []
    sensitivities = []
    
    for param_name, results in oat_results.items():
        if results.get('sensitivity') is not None:
            param_names.append(param_name)
            sensitivities.append(abs(results['sensitivity']))  # Use absolute values
    
    # Sort by sensitivity magnitude
    sorted_indices = np.argsort(sensitivities)[::-1]  # Descending order
    param_names = [param_names[i] for i in sorted_indices]
    sensitivities = [sensitivities[i] for i in sorted_indices]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars
    bars = ax.bar(param_names, sensitivities)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
              f'{height:.3f}', ha='center', va='bottom', fontsize=11)
    
    # Labels
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Absolute Sensitivity")
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Parameter Sensitivities for {output_type}")
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig, ax


def run_morris_analysis(proc, baseline_params, param_ranges=None, 
                       N=100, num_levels=4, output_type="final_X"):
    """
    Perform Morris method (elementary effects) sensitivity analysis.
    
    Args:
        proc (dict): Process configuration dictionary
        baseline_params (dict): Baseline parameter values (mu_max, Ks, Yxs)
        param_ranges (dict, optional): Dictionary specifying parameter ranges as fractions
                                      of baseline (e.g., {'mu_max': (0.8, 1.2)})
        N (int): Number of trajectories
        num_levels (int): Number of levels in parameter space grid
        output_type (str): Output metric to analyze
        
    Returns:
        tuple: (Si, Y_output, param_values)
            - Si: Morris sensitivity indices
            - Y_output: Model outputs for each parameter set
            - param_values: Parameter sets used for analysis
    """
    # Handle case where proc is a list of dictionaries
    if isinstance(proc, list):
        if len(proc) == 0:
            raise ValueError("Empty process list provided")
        proc = proc[0]  # Use the first process in the list
    
    # Extract baseline parameters
    mu_max_baseline = baseline_params["mu_max"]
    Ks_baseline = baseline_params["Ks"]
    Yxs_baseline = baseline_params["Yxs"]
    
    # Set default parameter ranges if not provided (±20% by default)
    if param_ranges is None:
        param_ranges = {
            'mu_max': (0.8, 1.2),  # ±20%
            'Ks': (0.8, 1.2),      # ±20%
            'Yxs': (0.8, 1.2)      # ±20%
        }
    
    # Define problem for SALib
    problem = {
        'num_vars': 3,
        'names': ['mu_max', 'Ks', 'Yxs'],
        'bounds': [
            [mu_max_baseline * param_ranges['mu_max'][0], 
             mu_max_baseline * param_ranges['mu_max'][1]],
            [Ks_baseline * param_ranges['Ks'][0], 
             Ks_baseline * param_ranges['Ks'][1]],
            [Yxs_baseline * param_ranges['Yxs'][0], 
             Yxs_baseline * param_ranges['Yxs'][1]]
        ]
    }
    
    # Generate parameter samples using Morris method
    param_values = morris.sample(problem, N=N, num_levels=num_levels)
    
    # Process configuration
    times = proc["time_points"]
    y0 = proc["y0"]
    feed_times = proc["feed_times"]
    Vb = proc["Vb"]
    S_feed = proc["S_feed"]
    
    # Run model for each parameter set
    Y_output = []
    
    for params in tqdm(param_values, desc="Running Morris simulations"):
        mu_max, Ks, Yxs = params
        
        try:
            S, X = simulate_monod(mu_max, Ks, Yxs, times, feed_times, Vb, S_feed, y0)
            
            # Calculate output metric based on selected type
            if output_type == "final_X":
                Y_output.append(float(X[-1]))  # Final metabolite concentration
            elif output_type == "max_X":
                Y_output.append(float(X.max()))  # Maximum metabolite concentration
            elif output_type == "final_S":
                Y_output.append(float(S[-1]))  # Final substrate concentration
            elif output_type == "yield":
                # Yield = X produced / S consumed
                X_produced = float(X[-1] - X[0])
                S_consumed = float(S[0] - S[-1])
                if S_consumed <= 0:
                    Y_output.append(0.0)
                else:
                    Y_output.append(X_produced / S_consumed)
            else:
                # Default to final metabolite
                Y_output.append(float(X[-1]))
                
        except Exception as e:
            print(f"Simulation failed: {e}")
            Y_output.append(np.nan)
    
    # Convert to numpy array
    Y_output = np.array(Y_output)
    
    # Handle NaN values (replace with mean to maintain trajectories)
    nan_indices = np.isnan(Y_output)
    if np.any(nan_indices):
        valid_mean = np.nanmean(Y_output)
        Y_output[nan_indices] = valid_mean
        print(f"Warning: {np.sum(nan_indices)} simulations failed and were replaced with mean value")
    
    # Perform Morris analysis
    Si = morris_analyze.analyze(problem, param_values, Y_output, print_to_console=False)
    
    return Si, Y_output, param_values


def plot_morris_indices(Si, problem, title=None, save_path=None):
    """
    Plot Morris sensitivity indices.
    
    Args:
        Si: Morris indices from SALib
        problem: Problem definition
        title (str, optional): Plot title
        save_path (str, optional): Path to save the figure
        
    Returns:
        tuple: (fig, axes) Matplotlib figure and axes
    """
    sns.set(style="whitegrid", context="notebook", font_scale=1.2)
    
    names = problem['names']
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Mean of absolute elementary effects (mu_star)
    axes[0].bar(names, Si['mu_star'])
    axes[0].set_title("Mean Absolute Elementary Effects (μ*)")
    axes[0].set_ylabel("μ*")
    axes[0].grid(True, alpha=0.3)
    
    # Standard deviation of elementary effects (sigma)
    axes[1].bar(names, Si['sigma'])
    axes[1].set_title("Standard Deviation of Elementary Effects (σ)")
    axes[1].set_ylabel("σ")
    axes[1].grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=16, y=1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig, axes


def plot_morris_scatter(Si, problem, title=None, save_path=None):
    """
    Create mu*-sigma scatter plot for Morris method results.
    
    Args:
        Si: Morris indices from SALib
        problem: Problem definition
        title (str, optional): Plot title
        save_path (str, optional): Path to save the figure
        
    Returns:
        tuple: (fig, ax) Matplotlib figure and axis
    """
    sns.set(style="whitegrid", context="notebook", font_scale=1.2)
    
    names = problem['names']
    mu_star = Si['mu_star']
    sigma = Si['sigma']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot scatter points
    ax.scatter(mu_star, sigma, s=80, alpha=0.8)
    
    # Add parameter names as labels
    for i, name in enumerate(names):
        ax.annotate(name, (mu_star[i], sigma[i]), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=12)
    
    # Add identity line
    max_val = max(max(mu_star), max(sigma)) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    
    # Labels
    ax.set_xlabel("Mean of Absolute Elementary Effects (μ*)")
    ax.set_ylabel("Standard Deviation of Elementary Effects (σ)")
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Morris Method: μ* vs σ")
    
    ax.grid(True, alpha=0.3)
    
    # Equal aspect ratio
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig, ax


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
    baseline_params = {
        "mu_max": mu_max_median,
        "Ks": Ks_process_5_median,
        "Yxs": Yxs_median,
    }
    
    # Example process
    proc = {
        "name": "process_5",
        "time_points": np.array([0, 1, 2, 3, 4, 5]),
        "feed_times": np.array([2.0, 4.0]),
        "Vb": 0.01,
        "S_feed": 10.0,
        "y0": np.array([1.0, 10.0, 1.0])
    }
    
    # Run Sobol analysis
    print("\n--- Running Sobol Sensitivity Analysis ---")
    Si, Y_output, param_values = run_sobol_analysis(
        proc, baseline_params, N=256, output_type="final_X")
    
    # Plot Sobol results
    fig1, axes1 = plot_sobol_indices(
        Si, 
        {"names": ["mu_max", "Ks", "Yxs"]}, 
        title="Sobol Sensitivity Analysis for Process 5"
    )
    
    # Plot second-order indices
    fig2, ax2 = plot_sobol_second_order(
        Si,
        {"names": ["mu_max", "Ks", "Yxs"]},
        title="Second-order Interactions"
    )
    
    # Run OAT analysis
    print("\n--- Running One-At-A-Time Sensitivity Analysis ---")
    oat_results, baseline_output = run_oat_analysis(
        proc, baseline_params, n_points=11, output_type="final_X")
    
    # Plot OAT results
    fig3, axes3 = plot_oat_results(
        oat_results, 
        baseline_params, 
        title="OAT Sensitivity Analysis for Process 5"
    )
    
    # Plot sensitivity comparison
    fig4, ax4 = plot_oat_sensitivity_comparison(
        oat_results,
        title="Parameter Sensitivity Comparison"
    )
    
    # Run Morris method
    print("\n--- Running Morris Method Sensitivity Analysis ---")
    Si_morris, Y_morris, params_morris = run_morris_analysis(
        proc, baseline_params, N=50, output_type="final_X")
    
    # Plot Morris results
    fig5, axes5 = plot_morris_indices(
        Si_morris,
        {"names": ["mu_max", "Ks", "Yxs"]},
        title="Morris Method Sensitivity Analysis"
    )
    
    # Plot Morris scatter
    fig6, ax6 = plot_morris_scatter(
        Si_morris,
        {"names": ["mu_max", "Ks", "Yxs"]},
        title="Morris Method: μ* vs σ"
    )
    
    plt.show()

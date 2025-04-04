"""
Cross-Validation for Monod Kinetic Model

This module implements leave-one-out cross-validation (LOOCV) for evaluating
the predictive performance of the Monod kinetic model. It uses parameters
estimated from N-1 processes to predict the behavior of the left-out process.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from model import *
from processes import processes

def perform_leave_one_out_cv(processes, parameter_data, medians=None):
    """
    Perform leave-one-out cross-validation for the Monod kinetic model.
    
    Args:
        processes (list): List of process dictionaries with experimental data
        parameter_data (dict): Dictionary of parameter samples from MCMC
                              (as loaded by load_parameter_file)
        medians (dict, optional): Pre-computed median parameter values
        
    Returns:
        tuple: (results_df, predicted_observed_list)
            - results_df: DataFrame with RMSE and R² metrics for each process
            - predicted_observed_list: List with observed and predicted values
    """
    # If medians not provided, compute them
    if medians is None:
        medians = _compute_medians(parameter_data)
    
    # Get median values for global parameters
    mu_max_median = float(np.median(parameter_data["mu_max"]))
    Yxs_median = float(np.median(parameter_data["Yxs"]))
    
    results = []
    predicted_observed = []
    
    # Loop through processes for LOOCV
    for proc in processes:
        proc_name = proc["name"]
        print(f"\n Cross-validating: leaving out {proc_name}")
        
        # Get process number
        proc_num = proc_name.split("_")[-1]
        
        # Exclude Ks from this process when computing the median
        Ks_values_remaining = []
        for other_name, other_values in parameter_data.items():
            # Match patterns like "Ks_process_1", "Ks_process_2", etc.
            if other_name.startswith("Ks_process_") and not other_name.endswith(f"_{proc_num}"):
                Ks_values_remaining.append(parameter_data[other_name])
        
        # Compute median Ks from remaining processes
        if Ks_values_remaining:
            Ks_cv_median = float(np.median(np.concatenate(Ks_values_remaining)))
        else:
            # Fallback to population mean if no process-specific values
            Ks_cv_median = float(np.median(parameter_data.get("Ks_mu", parameter_data.get("Ks", [0.0]))))
        
        # Extract process simulation settings
        y0 = proc["y0"]
        times = proc["time_points"]
        feed_times = proc["feed_times"]
        Vb = proc["Vb"]
        S_feed = proc["S_feed"]
        
        try:
            # Simulate with cross-validated parameters
            S_pred, X_pred = simulate_monod(
                mu_max=mu_max_median,
                Ks=Ks_cv_median,
                Yxs=Yxs_median,
                times=times,
                feed_times=feed_times,
                Vb=Vb,
                S_feed=S_feed,
                y0=y0
            )
            
            # Create masks for valid observations
            mask_X = ~jnp.isnan(proc["X"])
            mask_S = ~jnp.isnan(proc["S"])
            
            # Calculate performance metrics
            rmse_X = float(np.sqrt(mean_squared_error(proc["X"][mask_X], X_pred[mask_X])))
            rmse_S = float(np.sqrt(mean_squared_error(proc["S"][mask_S], S_pred[mask_S])))
            
            r2_X = float(r2_score(proc["X"][mask_X], X_pred[mask_X]))
            r2_S = float(r2_score(proc["S"][mask_S], S_pred[mask_S]))
            
            # Store results
            results.append({
                "process": proc_name,
                "rmse_X": rmse_X,
                "rmse_S": rmse_S,
                "r2_X": r2_X,
                "r2_S": r2_S,
                "Ks_used": Ks_cv_median
            })
            
            # Store predicted vs observed values
            predicted_observed.append({
                "process": proc_name,
                "X_obs": proc["X"][mask_X],
                "X_pred": X_pred[mask_X],
                "S_obs": proc["S"][mask_S],
                "S_pred": S_pred[mask_S],
                "times_X": proc["time_points"][mask_X],
                "times_S": proc["time_points"][mask_S]
            })
            
            print(f"→ Using Ks = {Ks_cv_median:.3f}")
            print(f"→ RMSE X: {rmse_X:.3f} | R² X: {r2_X:.3f}")
            print(f"→ RMSE S: {rmse_S:.3f} | R² S: {r2_S:.3f}")
            
        except Exception as e:
            print(f"Simulation failed for {proc_name}: {e}")
            results.append({
                "process": proc_name,
                "rmse_X": np.nan,
                "rmse_S": np.nan,
                "r2_X": np.nan,
                "r2_S": np.nan,
                "Ks_used": np.nan
            })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df, predicted_observed


def plot_cv_results(results_df, save_path=None):
    """
    Plot the cross-validation results as a bar chart.
    
    Args:
        results_df (DataFrame): DataFrame with RMSE and R² metrics
        save_path (str, optional): Path to save the plot
        
    Returns:
        Figure: Matplotlib figure object
    """
    sns.set(style="whitegrid", context="notebook", font_scale=1.2)
    
    # Prepare data for plotting (melt the DataFrame)
    plot_data = pd.melt(
        results_df, 
        id_vars=["process"], 
        value_vars=["rmse_X", "rmse_S", "r2_X", "r2_S"],
        var_name="Metric", 
        value_name="Value"
    )
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot RMSE values
    rmse_data = plot_data[plot_data["Metric"].isin(["rmse_X", "rmse_S"])]
    sns.barplot(x="process", y="Value", hue="Metric", data=rmse_data, ax=axes[0])
    axes[0].set_title("RMSE by Process")
    axes[0].set_xlabel("Process")
    axes[0].set_ylabel("RMSE (mmol/L)")
    axes[0].legend(title="")
    
    # Plot R² values
    r2_data = plot_data[plot_data["Metric"].isin(["r2_X", "r2_S"])]
    sns.barplot(x="process", y="Value", hue="Metric", data=r2_data, ax=axes[1])
    axes[1].set_title("R² by Process")
    axes[1].set_xlabel("Process")
    axes[1].set_ylabel("R²")
    axes[1].set_ylim(0, 1)  # R² should be between 0 and 1
    axes[1].legend(title="")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_cv_pred_vs_obs(predicted_observed, combined=True, save_dir=None):
    """
    Plot predicted vs observed values from cross-validation.
    
    Args:
        predicted_observed (list): List of dictionaries with predicted and observed values
        combined (bool): If True, combine all processes in one plot
        save_dir (str, optional): Directory to save plots
        
    Returns:
        tuple: (X_fig, S_fig) Matplotlib figure objects
    """
    sns.set(style="whitegrid", context="notebook", font_scale=1.2)
    
    if combined:
        # Combine all predicted vs. observed points
        X_obs_all, X_pred_all = [], []
        S_obs_all, S_pred_all = [], []
        process_labels_X, process_labels_S = [], []
        
        for entry in predicted_observed:
            X_obs = entry["X_obs"]
            X_pred = entry["X_pred"]
            S_obs = entry["S_obs"]
            S_pred = entry["S_pred"]
            
            X_obs_all.extend(X_obs)
            X_pred_all.extend(X_pred)
            S_obs_all.extend(S_obs)
            S_pred_all.extend(S_pred)
            
            process_labels_X.extend([entry["process"]] * len(X_obs))
            process_labels_S.extend([entry["process"]] * len(S_obs))
        
        # Convert to arrays
        X_obs_all = np.array(X_obs_all)
        X_pred_all = np.array(X_pred_all)
        S_obs_all = np.array(S_obs_all)
        S_pred_all = np.array(S_pred_all)
        
        # Create DataFrames for better plotting
        X_df = pd.DataFrame({
            "Observed": X_obs_all,
            "Predicted": X_pred_all,
            "Process": process_labels_X
        })
        
        S_df = pd.DataFrame({
            "Observed": S_obs_all,
            "Predicted": S_pred_all,
            "Process": process_labels_S
        })
        
        # Plot Metabolite (X)
        X_fig, X_ax = plt.subplots(figsize=(8, 7))
        sns.scatterplot(
            data=X_df,
            x="Observed",
            y="Predicted",
            hue="Process",
            alpha=0.7,
            s=60,
            ax=X_ax
        )
        
        # Add identity line
        lims = [
            min(X_ax.get_xlim()[0], X_ax.get_ylim()[0]),
            max(X_ax.get_xlim()[1], X_ax.get_ylim()[1])
        ]
        X_ax.plot(lims, lims, 'k--', alpha=0.7, zorder=0, label='Identity')
        X_ax.set_xlim(lims)
        X_ax.set_ylim(lims)
        
        # Add regression line
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(X_obs_all, X_pred_all)
        regression_label = f'Regression (R²={r_value**2:.3f})'
        x_line = np.linspace(min(X_obs_all), max(X_obs_all), 100)
        X_ax.plot(x_line, slope*x_line + intercept, 'r-', label=regression_label)
        
        X_ax.set_xlabel("Observed Metabolite X (mmol/L)")
        X_ax.set_ylabel("Predicted Metabolite X (mmol/L)")
        X_ax.set_title("Cross-Validation: Predicted vs Observed Metabolite X")
        X_ax.grid(True, alpha=0.3)
        X_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot Substrate (S)
        S_fig, S_ax = plt.subplots(figsize=(8, 7))
        sns.scatterplot(
            data=S_df,
            x="Observed",
            y="Predicted",
            hue="Process",
            alpha=0.7,
            s=60,
            ax=S_ax
        )
        
        # Add identity line
        lims = [
            min(S_ax.get_xlim()[0], S_ax.get_ylim()[0]),
            max(S_ax.get_xlim()[1], S_ax.get_ylim()[1])
        ]
        S_ax.plot(lims, lims, 'k--', alpha=0.7, zorder=0, label='Identity')
        S_ax.set_xlim(lims)
        S_ax.set_ylim(lims)
        
        # Add regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(S_obs_all, S_pred_all)
        regression_label = f'Regression (R²={r_value**2:.3f})'
        x_line = np.linspace(min(S_obs_all), max(S_obs_all), 100)
        S_ax.plot(x_line, slope*x_line + intercept, 'r-', label=regression_label)
        
        S_ax.set_xlabel("Observed Substrate S (mmol/L)")
        S_ax.set_ylabel("Predicted Substrate S (mmol/L)")
        S_ax.set_title("Cross-Validation: Predicted vs Observed Substrate S")
        S_ax.grid(True, alpha=0.3)
        S_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_dir:
            X_fig.savefig(f"{save_dir}/cv_X_pred_vs_obs.png", dpi=300, bbox_inches="tight")
            S_fig.savefig(f"{save_dir}/cv_S_pred_vs_obs.png", dpi=300, bbox_inches="tight")
        
        return X_fig, S_fig
    
    else:
        # Plot each process separately
        X_figs, S_figs = [], []
        
        for entry in predicted_observed:
            proc_name = entry["process"]
            X_obs = entry["X_obs"]
            X_pred = entry["X_pred"]
            S_obs = entry["S_obs"]
            S_pred = entry["S_pred"]
            
            # Plot Metabolite (X)
            X_fig, X_ax = plt.subplots(figsize=(6, 5))
            X_ax.scatter(X_obs, X_pred, alpha=0.7, edgecolor='k', s=60)
            
            # Add identity line
            lims = [
                min(X_ax.get_xlim()[0], X_ax.get_ylim()[0]),
                max(X_ax.get_xlim()[1], X_ax.get_ylim()[1])
            ]
            X_ax.plot(lims, lims, 'k--', alpha=0.7, zorder=0)
            X_ax.set_xlim(lims)
            X_ax.set_ylim(lims)
            
            X_ax.set_xlabel("Observed Metabolite X (mmol/L)")
            X_ax.set_ylabel("Predicted Metabolite X (mmol/L)")
            X_ax.set_title(f"Process {proc_name}: Predicted vs Observed Metabolite X")
            X_ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_dir:
                X_fig.savefig(f"{save_dir}/cv_{proc_name}_X_pred_vs_obs.png", dpi=300, bbox_inches="tight")
            
            X_figs.append(X_fig)
            
            # Plot Substrate (S)
            S_fig, S_ax = plt.subplots(figsize=(6, 5))
            S_ax.scatter(S_obs, S_pred, alpha=0.7, edgecolor='k', s=60)
            
            # Add identity line
            lims = [
                min(S_ax.get_xlim()[0], S_ax.get_ylim()[0]),
                max(S_ax.get_xlim()[1], S_ax.get_ylim()[1])
            ]
            S_ax.plot(lims, lims, 'k--', alpha=0.7, zorder=0)
            S_ax.set_xlim(lims)
            S_ax.set_ylim(lims)
            
            S_ax.set_xlabel("Observed Substrate S (mmol/L)")
            S_ax.set_ylabel("Predicted Substrate S (mmol/L)")
            S_ax.set_title(f"Process {proc_name}: Predicted vs Observed Substrate S")
            S_ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_dir:
                S_fig.savefig(f"{save_dir}/cv_{proc_name}_S_pred_vs_obs.png", dpi=300, bbox_inches="tight")
            
            S_figs.append(S_fig)
        
        return X_figs, S_figs


def plot_cv_time_series(predicted_observed, save_dir=None):
    """
    Plot time series of predicted vs observed values from cross-validation.
    
    Args:
        predicted_observed (list): List of dictionaries with predicted and observed values
        save_dir (str, optional): Directory to save plots
        
    Returns:
        list: List of figure objects
    """
    sns.set(style="whitegrid", context="notebook", font_scale=1.2)
    
    figures = []
    
    for entry in predicted_observed:
        proc_name = entry["process"]
        times_X = entry["times_X"]
        times_S = entry["times_S"]
        X_obs = entry["X_obs"]
        X_pred = entry["X_pred"]
        S_obs = entry["S_obs"]
        S_pred = entry["S_pred"]
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot Substrate (S)
        ax = axes[0]
        ax.plot(times_S, S_obs, 'o', label='Observed S', color='black', markersize=8)
        ax.plot(times_S, S_pred, '-', label='Predicted S (CV)', color='seagreen', linewidth=2)
        ax.set_ylabel("Substrate (mmol/L)")
        ax.set_title(f"Process {proc_name}: Cross-Validation Time Series")
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.3)
        
        # Calculate RMSE and R²
        rmse_S = np.sqrt(mean_squared_error(S_obs, S_pred))
        r2_S = r2_score(S_obs, S_pred)
        ax.text(0.05, 0.95, f"RMSE: {rmse_S:.3f}\nR²: {r2_S:.3f}",
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9))
        
        # Plot Metabolite (X)
        ax = axes[1]
        ax.plot(times_X, X_obs, 'o', label='Observed X', color='black', markersize=8)
        ax.plot(times_X, X_pred, '-', label='Predicted X (CV)', color='royalblue', linewidth=2)
        ax.set_xlabel("Time (hr)")
        ax.set_ylabel("Metabolite (mmol/L)")
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.3)
        
        # Calculate RMSE and R²
        rmse_X = np.sqrt(mean_squared_error(X_obs, X_pred))
        r2_X = r2_score(X_obs, X_pred)
        ax.text(0.05, 0.95, f"RMSE: {rmse_X:.3f}\nR²: {r2_X:.3f}",
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9))
        
        plt.tight_layout()
        
        if save_dir:
            fig.savefig(f"{save_dir}/cv_{proc_name}_time_series.png", dpi=300, bbox_inches="tight")
        
        figures.append(fig)
    
    return figures


def _compute_medians(parameter_data):
    """
    Internal helper function to compute median parameter values from the loaded parameter file.
    
    Args:
        parameter_data (dict): Dictionary of parameter samples as loaded by load_parameter_file
        
    Returns:
        dict: Dictionary of median parameter values
    """
    medians = {}
    
    # Global parameters
    parameter_keys = ["mu_max", "Ks_mu", "Ks_sigma", "Yxs", "Ks"]
    for param in parameter_keys:
        if param in parameter_data:
            medians[param] = float(np.median(parameter_data[param]))
    
    # Make sure we have Ks if Ks_mu is not available
    if "Ks" in medians and "Ks_mu" not in medians:
        medians["Ks_mu"] = medians["Ks"]
    
    # Process-specific parameters
    Ks_process_medians = {}
    for key in parameter_data.keys():
        if key.startswith("Ks_process_"):
            # Extract process number (e.g., "Ks_process_1" → "1")
            process_num = key.split("_")[-1]
            process_name = f"process_{process_num}"
            Ks_process_medians[process_name] = float(np.median(parameter_data[key]))
    
    medians["Ks_process"] = Ks_process_medians
    
    # Compute total Ks median across all processes if needed
    if not Ks_process_medians:
        return medians
        
    all_Ks_values = []
    for proc_values in Ks_process_medians.values():
        if isinstance(proc_values, (list, np.ndarray)):
            all_Ks_values.extend(proc_values)
        else:
            all_Ks_values.append(proc_values)
    
    if all_Ks_values:
        medians["Ks_total"] = float(np.median(all_Ks_values))
    
    return medians


# Example usage
if __name__ == "__main__":
    import re
    import numpy as np
    
    # Load parameters from file
    def load_parameter_file(file_path):
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace JAX Array format with numpy arrays for safer eval
        clean_content = re.sub(r'Array\((\[[\s\S]*?\]),\s*dtype=float32\)', r'np.array(\1)', content)
        
        # Evaluate safely with restricted namespace
        data = eval(clean_content, {"np": np})
        
        return data
    
    # Example parameter file path
    param_file = 'KS_param_all.txt'
    data = load_parameter_file(param_file)
    
    # Example processes definition (would be replaced with your actual data)
    processes = [
        {        "name": "process_4",
        "time_points": jnp.array([0, 12, 12.01, 24, 24.01, 36, 36.01, 48, 48.01,
                                  60, 60.01, 72, 72.01, 84, 84.01, 96, 96.01, 108, 108.01]),
        "S": jnp.array([1.9776049, 2.048203, 2.22420101, 1.8021382, 1.97881647,
                        1.6802096, 1.856361431, 1.0925149, 1.272586187,
                        0.50, 0.688753959, 0.16, 0.345575697,
                        jnp.nan, jnp.nan, 0.03, 0.217015832,
                        jnp.nan, jnp.nan]),
        "X": jnp.array([0.04, 0.12, 0.12, 0.35, 0.69,
                        0.69, 1.35, 1.35, 1.35,
                        2.15, 2.15, 2.55, 2.55,
                        2.95, 2.95, 3.06, 3.06,
                        3.23, 3.23]),
        "feed_times": jnp.array([12, 24, 36, 48, 60, 72, 84, 96, 108]),
        "Vb": 0.01,
        "S_feed": 20,
        "y0": jnp.array([1, 1.977, 0.04])
    },
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
    
    # Run cross-validation
    results_df, predicted_observed = perform_leave_one_out_cv(processes, data)
    
    # Plot results
    plot_cv_results(results_df)
    plot_cv_pred_vs_obs(predicted_observed, combined=True)
    plot_cv_time_series(predicted_observed)

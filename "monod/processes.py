"""
Processes Data Module

This module contains experimental process data for Monod kinetic model analysis.
Each process is a dictionary with experimental measurements and configuration.
"""

import jax.numpy as jnp

# Define processes dictionary
processes = [
    {
        "name": "process_1",
        "time_points": jnp.array([0, 12, 12.01, 24, 24.01, 36, 36.01, 48, 48.01,
                                  60, 60.01, 72, 72.01, 84, 84.01, 96, 96.01, 108, 108.01]),
        "S": jnp.array([0.95, 0.74, 1.001757019, 0.41, 0.682023469, 0.24, 0.504754744,
                        0.10, 0.372505251, 0.06, 0.322267943, 0.04, jnp.nan,
                        jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan]),
        "X": jnp.array([0.51, 0.96, 0.96, 1.31, 1.31, 1.84, 1.84, 2.31, 2.31,
                        2.49, 2.49, 2.93, 2.93, 3.21, 3.21, 3.45, 3.45, 3.84, 3.84]),
        "feed_times": jnp.array([12, 24, 36, 48, 60, 72, 84, 96, 108]),
        "Vb": 0.02,
        "S_feed": 15,
        "y0": jnp.array([1, 0.95, 0.51])
    },
    {
        "name": "process_2",
        "time_points": jnp.array([0, 12, 12.01, 24, 24.01, 36, 36.01, 48, 48.01,
                                  60, 60.01, 72, 72.01, 84, 84.01, 96, 96.01, 108, 108.01]),
        "S": jnp.array([4.934649, 4.766027, 4.826479274, 4.276312, 4.338460964,
                        3.0775678, 3.144191549, 1.0423965, 1.116740043,
                        jnp.nan, jnp.nan, jnp.nan, jnp.nan,
                        0.042221755, jnp.nan, jnp.nan, jnp.nan,
                        jnp.nan, jnp.nan]),
        "X": jnp.array([0.21352927, 0.36417016, 0.36417016, 0.9871855, 0.9871855,
                        2.5310051, 2.5310051, 4.426204, 4.426204,
                        5.4631715, 5.4631715, 5.416769, 5.416769,
                        5.4013944, 5.4013944, 5.729346, 5.729346,
                        5.5023885, 5.5023885]),
        "feed_times": jnp.array([12, 24, 36, 48, 60, 72, 84, 96, 108]),
        "Vb": 0.004,
        "S_feed": 20,
        "y0": jnp.array([1, 4.93, 0.213])
    },
    {
        "name": "process_3",
        "time_points": jnp.array([0, 12, 12.01, 24, 24.01, 36, 36.01, 48, 48.01,
                                  60, 60.01, 72, 72.01, 84, 84.01, 96, 96.01, 108, 108.01]),
        "S": jnp.array([5.6556683, 5.3213196, 5.359727062, 5.3271165, 5.365349241,
                        5.11142, 5.150351417, 4.348259, 4.390030533,
                        2.8362167, 2.883731479, 0.391334, 0.448177058,
                        0.06582995, 0.12371433, 0.08515732, 0.142743585,
                        0.040596463, 0.09813263]),
        "X": jnp.array([0.12772582, 0.10089108, 0.10089108, 0.49272382, 0.49272382,
                        0.5631874, 0.5631874, 1.3316774, 1.3316774,
                        2.82884, 2.82884, 5.437546, 5.437546,
                        5.816827, 5.816827, 5.7353964, 5.7353964,
                        5.9404325, 5.9404325]),
        "feed_times": jnp.array([12, 24, 36, 48, 60, 72, 84, 96, 108]),
        "Vb": 0.004,
        "S_feed": 15,
        "y0": jnp.array([1, 5.65, 0.127])
    },
    {
        "name": "process_4",
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
        "y0": jnp.array([1, 1.0222119, 0.08])
    }
]

# Utility functions for accessing processes
def get_process_by_name(name):
    """
    Retrieve a process dictionary by its name.
    
    Args:
        name (str): Name of the process (e.g., 'process_1')
    
    Returns:
        dict: Process configuration dictionary
    
    Raises:
        ValueError: If no process with the given name is found
    """
    for process in processes:
        if process['name'] == name:
            return process
    raise ValueError(f"No process found with name {name}")

def get_all_process_names():
    """
    Get a list of all process names.
    
    Returns:
        list: List of process names
    """
    return [process['name'] for process in processes]

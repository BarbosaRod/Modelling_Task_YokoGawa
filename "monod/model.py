"""
Monod ODE Kinetic Model with Fed-Batch Simulation

This module implements a ODE Monod kinetic model for substrate utilization
The model simulates substrate consumption S  and metabolite production X in a fed-batch process.

Key components:
- Monod kinetics for growth rate calculation
- Fed-batch feeding implementation
- Differential equation solver using JAX and Diffrax

Variables:
- X: Metabolite concentration (mmol/L)
- S: Substrate concentration (mmol/L)
- V: Reaction volume (L)
- time: Simulation time (hr) - proposed units

Parameters:
- μ_max: Maximum specific reaction rate (hr⁻¹)
- Ks: Substrate affinity constant (mmol/L)
- Yxs: Yield coefficient (mmol metabolite / mmol substrate)
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, Tsit5, SaveAt, ODETerm, PIDController
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def monod_ode(t, y, args):
    """
    Monod model ordinary differential equations without inhibition.
    
    Args:
        t (float): Current time point (hr)
        y (array): State vector [V, S, X]
            - V: Reaction volume (L)
            - S: Substrate concentration (mmol/L)
            - X: Metabolite concentration (mmol/L)
        args (tuple): Model parameters (μ_max, Ks, Yxs)
            - μ_max: Maximum specific reaction rate (hr⁻¹)
            - Ks: Substrate affinity constant (mmol/L)
            - Yxs: Yield coefficient (mmol metabolite / mmol substrate)
    
    Returns:
        array: Derivatives [dV/dt, dS/dt, dX/dt]
    """
    mu_max, Ks, Yxs = args
    V, S, X = y
    
    # Calculate specific growth rate using Monod equation
    mu = mu_max * S / (Ks + S)
    
    # Calculate derivatives
    dXdt = mu * X  # Metabolite production
    dSdt = -1 / Yxs * dXdt  # Substrate consumption
    dVdt = 0.0  # Volume doesn't change between feedings
    
    return jnp.array([dVdt, dSdt, dXdt])


def apply_feed(t, y, feed_times, Vb, S_feed):
    """
    Apply substrate feeding at specified time points.
    
    Args:
        t (float): Current time point (hr)
        y (array): Current state vector [V, S, X]
        feed_times (array): Array of feeding time points (hr)
        Vb (float): Volume of feed added at each feeding (L)
        S_feed (float): Substrate concentration in feed (mmol/L)
    
    Returns:
        array: Updated state vector after feeding
    """
    # Check if current time is close to any feeding time
    close = jnp.any(jnp.isclose(t, feed_times, atol=1e-3))
    
    def fed_state():

        """Calculate new state after feeding."""

        V, S, X = y
        
        # Update volume
        V_new = V + Vb
        
        # Update substrate concentration (mass balance)
        S_new = (S * V + S_feed * Vb) / V_new
        
        # Update metabolite concentration (dilution effect)
        X_new = X * V / V_new
        
        return jnp.array([V_new, S_new, X_new])
    
    # Conditionally apply feeding if time matches a feed time
    return jax.lax.cond(close, lambda y: fed_state(), lambda y: y, y)


def simulate_monod(mu_max, Ks, Yxs, times, feed_times, Vb, S_feed, y0):
    """
    Simulate the Monod model with fed-batch operation.
    
    Args:
        mu_max (float): Maximum specific reaction rate (hr⁻¹)
        Ks (float): Substrate affinity constant (mmol/L)
        Yxs (float): Yield coefficient (mmol metabolite / mmol substrate)
        times (array): Time points for simulation (hr)
        feed_times (array): Time points for substrate feeding (hr)
        Vb (float): Volume of feed added at each feeding (L)
        S_feed (float): Substrate concentration in feed (mmol/L)
        y0 (array): Initial state vector [V₀, S₀, X₀]
    
    Returns:
        tuple: Arrays of substrate and biomass concentrations at each time point
    """
    args = (mu_max, Ks, Yxs)
    term = ODETerm(monod_ode)
    
    # Configure numerical integration controller
    controller = PIDController(rtol=1e-3, atol=1e-4)

    def step_with_feed(t0, t1, y):

        """Simulate one time step with potential feeding event."""
        
        # First check and apply feeding if needed
        y = apply_feed(t0, y, feed_times, Vb, S_feed)
        
        # Then solve ODEs for this time step
        sol = diffeqsolve(
            term, 
            solver=Tsit5(),  # 5th order Tsitouras method
            t0=t0, 
            t1=t1, 
            dt0=0.1,  # Initial step size
            y0=y, 
            args=args, 
            saveat=SaveAt(t1=True),  # Save only at t1
            stepsize_controller=controller
        )
        return sol.ys[-1]  # Return final state

    # Run simulation step by step
    y = y0
    ys = [y0]
    for i in range(len(times) - 1):
        y = step_with_feed(times[i], times[i + 1], y)
        ys.append(y)

    # Collect results
    ys = jnp.stack(ys)
    
    # Return substrate and metabolite concentrations
    return ys[:, 1], ys[:, 2]  # S, X










##################################################################################################################################################



"""
Example usage of the Monod kinetic model for substrate-to-metabolite conversion.

This script demonstrates how to:
1. Set up model parameters
2. Configure simulation conditions
3. Run the model simulation
4. Visualize results
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
jax.config.update('jax_platform_name', 'cpu')  # Use CPU for deterministic results
key = jax.random.PRNGKey(42)

# -------------- Set Model Parameters --------------
# Kinetic parameters
mu_max = 0.5      # Maximum specific reaction rate (hr⁻¹)
Ks = 10.0         # Substrate affinity constant (mmol/L)
Yxs = 0.8         # Yield coefficient (mmol metabolite / mmol substrate)

# Initial conditions
V0 = 1.0          # Initial volume (L)
S0 = 20.0         # Initial substrate concentration (mmol/L)
X0 = 1.0          # Initial metabolite concentration (mmol/L)
y0 = jnp.array([V0, S0, X0])

# Feed conditions
Vb = 0.1          # Volume of each feed (L)
S_feed = 100.0    # Substrate concentration in feed (mmol/L)

# -------------- Set Simulation Time --------------
# Total simulation time
t_end = 24.0     # hours
dt = 0.5         # Time step for output
times = jnp.arange(0.0, t_end + dt, dt)

# Define feeding times (hours)
feed_times = jnp.array([6.0, 12.0, 18.0])

# -------------- Run Simulation --------------
S, X = simulate_monod(mu_max, Ks, Yxs, times, feed_times, Vb, S_feed, y0)

# -------------- Visualize Results --------------
plt.figure(figsize=(10, 6))

# Plot substrate and metabolite concentrations
plt.subplot(2, 1, 1)
plt.plot(times, S, 'b-', label='Substrate (S)')
plt.plot(times, X, 'r-', label='Metabolite (X)')
plt.ylabel('Concentration (mmol/L)')
plt.grid(True, alpha=0.3)
plt.legend()

# Mark feeding times with vertical lines
for ft in feed_times:
    plt.axvline(x=ft, color='green', linestyle='--', alpha=0.5)

# Add second subplot for conversion rate
plt.subplot(2, 1, 2)
# Calculate specific reaction rate at each time point
reaction_rate = mu_max * S / (Ks + S)
plt.plot(times, reaction_rate, 'g-')
plt.xlabel('Time (hr)')
plt.ylabel('Reaction Rate (hr⁻¹)')
plt.grid(True, alpha=0.3)

# Mark feeding times with vertical lines
for ft in feed_times:
    plt.axvline(x=ft, color='green', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.suptitle('Monod Kinetic Model Simulation', y=1.02)
plt.savefig('monod_simulation_results.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------- Print Summary Statistics --------------
print(f"Initial substrate concentration: {S[0]:.2f} mmol/L")
print(f"Final substrate concentration: {S[-1]:.2f} mmol/L")
print(f"Initial metabolite concentration: {X[0]:.2f} mmol/L")
print(f"Final metabolite concentration: {X[-1]:.2f} mmol/L")
print(f"Substrate consumption: {S[0] - S[-1]:.2f} mmol/L")
print(f"Metabolite production: {X[-1] - X[0]:.2f} mmol/L")
print(f"Overall yield: {(X[-1] - X[0])/(S[0] - S[-1]):.4f} mmol metabolite/mmol substrate")

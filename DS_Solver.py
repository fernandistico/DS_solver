#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:35:06 2024

Dinamic systems solver

@author: fvera
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the Van der Pol Oscillator differential equations
def van_der_pol(t, state, mu):
    """
    Computes the derivatives for the Van der Pol oscillator.

    Parameters:
        t (float): Time variable (not used explicitly as the system is autonomous).
        state (list): Current state vector [x, y].
        mu (float): Parameter controlling nonlinearity and damping.

    Returns:
        list: Derivatives [dx/dt, dy/dt].
    """
    x, y = state
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return [dxdt, dydt]

# Define the FitzHugh-Nagumo model differential equations
def fitzhugh_nagumo(t, state, I):
    """
    Computes the derivatives for the FitzHugh-Nagumo model.

    Parameters:
        t (float): Time variable.
        state (list): Current state vector [v, w].
        I (float): External current input.

    Returns:
        list: Derivatives [dv/dt, dw/dt].
    """
    v, w = state
    dvdt = v - (v**3) / 3 - w + I
    dwdt = 0.08 * (v + 0.7 - 0.8 * w)
    return [dvdt, dwdt]

# Parameters for the Van der Pol oscillator
mu = 1.0  # Nonlinearity/damping parameter; vary to observe different behaviors
initial_state_vdp = [0.5, 0.5]  # Initial conditions [x0, y0]

# Parameters for the FitzHugh-Nagumo model
I = 0.5   # External current; vary to observe different dynamics
initial_state_fhn = [0.0, 0.0]  # Initial conditions [v0, w0]

# Time span and evaluation points
t_span = (0, 100)  # Start and end times
t_eval = np.linspace(*t_span, 10000)  # Time points where the solution is computed

# Solve the Van der Pol oscillator equations
sol_vdp = solve_ivp(
    fun=lambda t, y: van_der_pol(t, y, mu),
    t_span=t_span,
    y0=initial_state_vdp,
    t_eval=t_eval,
    method='RK45'
)

# Solve the FitzHugh-Nagumo equations
sol_fhn = solve_ivp(
    fun=lambda t, y: fitzhugh_nagumo(t, y, I),
    t_span=t_span,
    y0=initial_state_fhn,
    t_eval=t_eval,
    method='RK45'
)

# Function to plot phase portraits
def plot_phase_portrait(sol, title, xlabel, ylabel):
    """
    Plots the phase portrait of a dynamical system.

    Parameters:
        sol (OdeResult): Solution object returned by solve_ivp.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(sol.y[0], sol.y[1], color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot the phase portrait of the Van der Pol oscillator
plot_phase_portrait(
    sol_vdp,
    title='Limit Cycle of the Van der Pol Oscillator',
    xlabel='x',
    ylabel='y'
)

# Plot the phase portrait of the FitzHugh-Nagumo model
plot_phase_portrait(
    sol_fhn,
    title='Limit Cycle of the FitzHugh-Nagumo Model',
    xlabel='v',
    ylabel='w'
)

# Function to plot time series
def plot_time_series(sol, variables, labels, title):
    """
    Plots the time series of variables from the solution.

    Parameters:
        sol (OdeResult): Solution object returned by solve_ivp.
        variables (list): List of variable indices to plot.
        labels (list): Corresponding labels for the variables.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 5))
    for idx, label in zip(variables, labels):
        plt.plot(sol.t, sol.y[idx], label=label)
    plt.title(title)
    plt.xlabel('Time t')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Optional: Plot time series for the Van der Pol oscillator
plot_time_series(
    sol_vdp,
    variables=[0, 1],
    labels=['x(t)', 'y(t)'],
    title='Van der Pol Oscillator Variables over Time'
)

# Optional: Plot time series for the FitzHugh-Nagumo model
plot_time_series(
    sol_fhn,
    variables=[0, 1],
    labels=['v(t)', 'w(t)'],
    title='FitzHugh-Nagumo Model Variables over Time'
)

# Example: Vary mu in the Van der Pol oscillator
mu_values = [0.5, 1.0, 2.0]
plt.figure(figsize=(10, 5))
for mu in mu_values:
    sol = solve_ivp(
        fun=lambda t, y: van_der_pol(t, y, mu),
        t_span=t_span,
        y0=initial_state_vdp,
        t_eval=t_eval,
        method='RK45'
    )
    plt.plot(sol.y[0], sol.y[1], label=f'μ = {mu}')

plt.title('Effect of μ on the Van der Pol Oscillator')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Example: Vary I in the FitzHugh-Nagumo model
I_values = [0.5, 0.7, 0.9]
plt.figure(figsize=(10, 5))
for I in I_values:
    sol = solve_ivp(
        fun=lambda t, y: fitzhugh_nagumo(t, y, I),
        t_span=t_span,
        y0=initial_state_fhn,
        t_eval=t_eval,
        method='RK45'
    )
    plt.plot(sol.y[0], sol.y[1], label=f'I = {I}')

plt.title('Effect of I on the FitzHugh-Nagumo Model')
plt.xlabel('v')
plt.ylabel('w')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

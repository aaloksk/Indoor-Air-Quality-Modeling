# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:43:25 2024

@author: Aalok Sharma Kafle
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
room_volume = 10 * 10 * 3  # m³ (Length x Width x Height)
air_exchange_rate = 5  # exchanges per hour
lpg_emission_nox = 0.2  # g/s (NOx emission rate)
lpg_emission_sox = 0.0002  # g/s (SOx emission rate)
operation_hours = 12  # hours/day
outdoor_nox = 50  # ppb
outdoor_sox = 0.5  # ppb
nox_half_life = 50 / 60  # hours
sox_half_life = 1 + 15 / 60  # hours

# Conversion factors
ppb_to_grams_per_cubic_meter = lambda ppb, mol_weight: ppb * mol_weight / (24.45 * 10**9)
nox_mol_weight = 46  # g/mol for NO2
sox_mol_weight = 64  # g/mol for SO2
outdoor_nox_conc = ppb_to_grams_per_cubic_meter(outdoor_nox, nox_mol_weight)
outdoor_sox_conc = ppb_to_grams_per_cubic_meter(outdoor_sox, sox_mol_weight)

# Decay constants
k_nox = np.log(2) / (nox_half_life * 3600)  # per second
k_sox = np.log(2) / (sox_half_life * 3600)  # per second

# Ventilation rate in per second
ventilation_rate = air_exchange_rate / 3600  # exchanges per second

# Emission rates in g/m³/s
nox_emission_rate = lpg_emission_nox / room_volume  # g/m³/s
sox_emission_rate = lpg_emission_sox / room_volume  # g/m³/s

# Define the ODE system
def pollutant_ode(t, y):
    """
    ODE system for NOx and SOx concentrations.
    """
    nox, sox = y
    if 10 <= (t / 3600) % 24 < 22:  # Operating hours
        nox_emission = nox_emission_rate
        sox_emission = sox_emission_rate
    else:  # Non-operating hours
        nox_emission = 0
        sox_emission = 0

    # Differential equations
    d_nox_dt = nox_emission - ventilation_rate * (nox - outdoor_nox_conc) - k_nox * nox
    d_sox_dt = sox_emission - ventilation_rate * (sox - outdoor_sox_conc) - k_sox * sox
    return [d_nox_dt, d_sox_dt]

# Initial conditions
initial_conditions = [outdoor_nox_conc, outdoor_sox_conc]  # Start with outdoor air concentrations

# Time span (24 hours in seconds)
t_span = (0, 24 * 3600)  # seconds
t_eval = np.linspace(0, 24 * 3600, 1000)  # Evaluate at 1000 points for smooth curves

# Solve the ODEs
solution = solve_ivp(pollutant_ode, t_span, initial_conditions, t_eval=t_eval, method='RK45')

# Extract solutions
time_hours = solution.t / 3600  # Convert time to hours
nox_concentration = solution.y[0]
sox_concentration = solution.y[1]

# Convert concentrations to ppb
nox_concentration_ppb = nox_concentration / ppb_to_grams_per_cubic_meter(1, nox_mol_weight)
sox_concentration_ppb = sox_concentration / ppb_to_grams_per_cubic_meter(1, sox_mol_weight)

# Plot results
plt.figure(figsize=(14, 6))

# NOx Plot
plt.subplot(1, 2, 1)
plt.plot(time_hours, nox_concentration_ppb, label='NOx Concentration (ppb)')
plt.axhline(outdoor_nox, color='r', linestyle='--', label='Outdoor NOx (50 ppb)')
plt.xlabel('Time (hours)')
plt.ylabel('Concentration (ppb)')
plt.title('NOx Concentration Over 24 Hours')
plt.legend()

# SOx Plot
plt.subplot(1, 2, 2)
plt.plot(time_hours, sox_concentration_ppb, label='SOx Concentration (ppb)', color='orange')
plt.axhline(outdoor_sox, color='r', linestyle='--', label='Outdoor SOx (0.5 ppb)')
plt.xlabel('Time (hours)')
plt.ylabel('Concentration (ppb)')
plt.title('SOx Concentration Over 24 Hours')
plt.legend()

plt.tight_layout()
plt.show()




#PROBLEM 2

# Define customer data for hourly scaling
customer_counts = [20, 50, 80, 50, 40, 20, 30, 120, 100, 70, 40, 30]  # Customers per hour
baseline_customers = 50  # Base for scaling

# Scale emission rates based on customer count
scaled_nox_emission_rates = [(count / baseline_customers) * nox_emission_rate for count in customer_counts]
scaled_sox_emission_rates = [(count / baseline_customers) * sox_emission_rate for count in customer_counts]

# Update the ODE system to include time-dependent emission rates
def pollutant_ode_customers(t, y):
    """
    ODE system for NOx and SOx concentrations with customer-dependent emission rates.
    """
    nox, sox = y
    hour = int((t / 3600) % 24)  # Current hour in 24-hour format
    
    if 10 <= hour < 22:  # Operating hours
        hour_index = hour - 10
        nox_emission = scaled_nox_emission_rates[hour_index]
        sox_emission = scaled_sox_emission_rates[hour_index]
    else:  # Non-operating hours
        nox_emission = 0
        sox_emission = 0

    # Differential equations
    d_nox_dt = nox_emission - ventilation_rate * (nox - outdoor_nox_conc) - k_nox * nox
    d_sox_dt = sox_emission - ventilation_rate * (sox - outdoor_sox_conc) - k_sox * sox
    return [d_nox_dt, d_sox_dt]

# Solve the ODEs with updated emission rates
solution_customers = solve_ivp(
    pollutant_ode_customers, t_span, initial_conditions, t_eval=t_eval, method='RK45'
)

# Extract solutions
time_hours_customers = solution_customers.t / 3600  # Convert time to hours
nox_concentration_customers = solution_customers.y[0]
sox_concentration_customers = solution_customers.y[1]

# Convert concentrations to ppb
nox_concentration_ppb_customers = nox_concentration_customers / ppb_to_grams_per_cubic_meter(1, nox_mol_weight)
sox_concentration_ppb_customers = sox_concentration_customers / ppb_to_grams_per_cubic_meter(1, sox_mol_weight)

# Plot results
plt.figure(figsize=(14, 6))

# NOx Plot
plt.subplot(1, 2, 1)
plt.plot(time_hours_customers, nox_concentration_ppb_customers, label='NOx Concentration (ppb)')
plt.axhline(outdoor_nox, color='r', linestyle='--', label='Outdoor NOx (50 ppb)')
plt.xlabel('Time (hours)')
plt.ylabel('Concentration (ppb)')
plt.title('NOx Concentration (Customer-Dependent)')
plt.legend()

# SOx Plot
plt.subplot(1, 2, 2)
plt.plot(time_hours_customers, sox_concentration_ppb_customers, label='SOx Concentration (ppb)', color='orange')
plt.axhline(outdoor_sox, color='r', linestyle='--', label='Outdoor SOx (0.5 ppb)')
plt.xlabel('Time (hours)')
plt.ylabel('Concentration (ppb)')
plt.title('SOx Concentration (Customer-Dependent)')
plt.legend()

plt.tight_layout()
plt.show()


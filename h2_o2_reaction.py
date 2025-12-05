"""
H2 + O2 -> H2O Reaction Kinetics Simulation
============================================

This script models the kinetic reaction of hydrogen and oxygen to produce water.
It calculates and plots:
- Concentration of reactants (H2, O2) and products (H2O) over time
- Reaction rate over time

Requires: cantera >= 3.0, matplotlib >= 2.0, numpy

.. tags:: Python, combustion, reaction kinetics, transient simulation
"""

import numpy as np
import matplotlib.pyplot as plt
import cantera as ct

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

# Temperature and pressure conditions
T_initial = 1000.0  # Initial temperature [K]
p = 1.1 * ct.one_atm  # Pressure [Pa]

# Time integration parameters
t_end = 0.005  # Final time [s]
n_steps = 1000  # Number of time steps

# Initial composition (mole fractions)
# H2:1, O2:0.5 (stoichiometric ratio for H2 + 0.5*O2 -> H2O)
initial_composition = 'H2:1.0, O2:0.5, N2:3.5'  # N2 as inert diluent

# ============================================================================
# SETUP
# ============================================================================

# Create a solution object using GRI-Mech 3.0 (includes H2 combustion mechanism)
gas = ct.Solution('gri30.yaml')

# Set initial state
gas.TPX = T_initial, p, initial_composition

print("=" * 70)
print("H2 + O2 → H2O REACTION KINETICS SIMULATION")
print("=" * 70)
print(f"Initial Temperature: {T_initial} K")
print(f"Pressure: {p / ct.one_atm:.2f} atm")
print(f"Initial Composition: {initial_composition}")
print(f"Simulation Time: 0 to {t_end} s")
print("=" * 70)

# Get species indices for H2, O2, and H2O
i_h2 = gas.species_index('H2')
i_o2 = gas.species_index('O2')
i_h2o = gas.species_index('H2O')

# ============================================================================
# TIME INTEGRATION USING IdealGasReactor
# ============================================================================

# Create a reactor
reactor = ct.IdealGasReactor(gas)
reactor.volume = 1.0  # Volume [m³] (doesn't affect homogeneous kinetics)

# Create a reactor network
network = ct.ReactorNet([reactor])

# Time array for simulation
times = np.linspace(0, t_end, n_steps)

# Storage arrays for results
concentrations_h2 = np.zeros(n_steps)
concentrations_o2 = np.zeros(n_steps)
concentrations_h2o = np.zeros(n_steps)
temperatures = np.zeros(n_steps)
reaction_rates = np.zeros(n_steps)
mole_fractions_h2 = np.zeros(n_steps)
mole_fractions_o2 = np.zeros(n_steps)
mole_fractions_h2o = np.zeros(n_steps)

# Run the simulation
print("\nRunning transient reaction simulation...")
for i, t in enumerate(times):
    network.advance(t)
    
    # Store concentrations in kmol/m³
    concentrations_h2[i] = reactor.thermo.concentrations[i_h2]
    concentrations_o2[i] = reactor.thermo.concentrations[i_o2]
    concentrations_h2o[i] = reactor.thermo.concentrations[i_h2o]
    temperatures[i] = reactor.T
    
    # Calculate reaction rate for H2O formation
    # Rate = mass production rate / molecular weight
    # Units: kmol/(m³·s)
    h2o_rate = reactor.thermo.creation_rates[i_h2o]
    reaction_rates[i] = h2o_rate

    # Store mole fractions
    mole_fractions_h2[i] = reactor.thermo.X[i_h2]
    mole_fractions_o2[i] = reactor.thermo.X[i_o2]
    mole_fractions_h2o[i] = reactor.thermo.X[i_h2o]
    
    if i % 50 == 0:
        print(f"  Time: {t:.4f} s, T: {reactor.T:.2f} K, "
              f"[H2]: {concentrations_h2[i]:.6e}, "
              f"[O2]: {concentrations_o2[i]:.6e}, "
              f"[H2O]: {concentrations_h2o[i]:.6e} kmol/m³")

print("Simulation complete!")

# ============================================================================
# PLOTTING
# ============================================================================

# Create a figure with multiple subplots
fig = plt.figure(figsize=(14, 10))

# Plot 1: Concentration vs Time
ax1 = plt.subplot(2, 2, 1)
ax1.plot(times * 1000, concentrations_h2 * 1e6, 'b-', linewidth=2, label='H₂')
ax1.plot(times * 1000, concentrations_o2 * 1e6, 'r-', linewidth=2, label='O₂')
ax1.plot(times * 1000, concentrations_h2o * 1e6, 'g-', linewidth=2, label='H₂O')
ax1.set_xlabel('Time [ms]', fontsize=12)
ax1.set_ylabel('Concentration [μmol/m³]', fontsize=12)
ax1.set_title('Species Concentrations vs Time', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Mole Fractions vs Time
ax2 = plt.subplot(2, 2, 2)
ax2.plot(times * 1000, mole_fractions_h2 * 100, 'b-', linewidth=2, label='H₂')
ax2.plot(times * 1000, mole_fractions_o2 * 100, 'r-', linewidth=2, label='O₂')
ax2.plot(times * 1000, mole_fractions_h2o * 100, 'g-', linewidth=2, label='H₂O')
ax2.set_xlabel('Time [ms]', fontsize=12)
ax2.set_ylabel('Mole Fraction [%]', fontsize=12)
ax2.set_title('Species Mole Fractions vs Time', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Plot 3: H2O Production Rate vs Time
ax3 = plt.subplot(2, 2, 3)
ax3.plot(times * 1000, reaction_rates * 1e6, 'g-', linewidth=2)
ax3.set_xlabel('Time [ms]', fontsize=12)
ax3.set_ylabel('H₂O Production Rate [μmol/(m³·s)]', fontsize=12)
ax3.set_title('H₂O Formation Rate vs Time', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Temperature vs Time
ax4 = plt.subplot(2, 2, 4)
ax4.plot(times * 1000, temperatures, 'orange', linewidth=2)
ax4.set_xlabel('Time [ms]', fontsize=12)
ax4.set_ylabel('Temperature [K]', fontsize=12)
ax4.set_title('Temperature vs Time', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('h2_o2_reaction_analysis.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'h2_o2_reaction_analysis.png'")
plt.show()

# ============================================================================
# ADDITIONAL ANALYSIS PLOT: Comparative View
# ============================================================================

fig2, ax = plt.subplots(figsize=(12, 6))

# Normalize concentrations to initial values for comparison
initial_conc_h2 = concentrations_h2[0]
initial_conc_o2 = concentrations_o2[0]

normalized_h2 = (concentrations_h2 / initial_conc_h2) * 100 if initial_conc_h2 > 0 else concentrations_h2
normalized_o2 = (concentrations_o2 / initial_conc_o2) * 100 if initial_conc_o2 > 0 else concentrations_o2

ax.plot(times * 1000, normalized_h2, 'b-', linewidth=2.5, label='H₂ (% of initial)')
ax.plot(times * 1000, normalized_o2, 'r-', linewidth=2.5, label='O₂ (% of initial)')
ax.plot(times * 1000, (concentrations_h2o - concentrations_h2o[0]) / initial_conc_h2 * 100, 
        'g-', linewidth=2.5, label='H₂O formation')
ax.set_xlabel('Time [ms]', fontsize=12)
ax.set_ylabel('Normalized Concentration [%]', fontsize=12)
ax.set_title('H₂ + O₂ → H₂O Reaction Progress', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('h2_o2_reaction_normalized.png', dpi=300, bbox_inches='tight')
print("Normalized plot saved as 'h2_o2_reaction_normalized.png'")
plt.show()

# ============================================================================
# PRINT SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 70)
print("SIMULATION SUMMARY")
print("=" * 70)
print(f"Final Temperature: {temperatures[-1]:.2f} K")
print(f"Temperature Change: {temperatures[-1] - temperatures[0]:.2f} K")
print(f"\nInitial Concentrations:")
print(f"  H₂: {concentrations_h2[0]:.6e} kmol/m³")
print(f"  O₂: {concentrations_o2[0]:.6e} kmol/m³")
print(f"  H₂O: {concentrations_h2o[0]:.6e} kmol/m³")
print(f"\nFinal Concentrations:")
print(f"  H₂: {concentrations_h2[-1]:.6e} kmol/m³ ({(1 - concentrations_h2[-1]/concentrations_h2[0])*100:.2f}% consumed)")
print(f"  O₂: {concentrations_o2[-1]:.6e} kmol/m³ ({(1 - concentrations_o2[-1]/concentrations_o2[0])*100:.2f}% consumed)")
print(f"  H₂O: {concentrations_h2o[-1]:.6e} kmol/m³")
print(f"\nMax H₂O Production Rate: {np.max(reaction_rates):.6e} kmol/(m³·s)")
print(f"Average H₂O Production Rate: {np.mean(reaction_rates):.6e} kmol/(m³·s)")
print("=" * 70)

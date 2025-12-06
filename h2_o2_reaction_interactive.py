"""
Interactive H2 + O2 -> H2O Reaction Kinetics Simulator
======================================================
Nathan Bryce | MEGN566
Code written with help from GitHub Copilot


Web-based interactive tool for exploring the H2 + O2 → H2O reaction kinetics
with adjustable temperature and pressure parameters.

Requires: cantera >= 3.0, streamlit >= 1.0, matplotlib >= 2.0, numpy

Run with: streamlit run h2_o2_reaction_interactive.py

"""

import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
import streamlit as st

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="CH4 + O2 Stoichiometric Combustion Simulator",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("CH₄ + 2 O₂ → CO₂ + 2 H₂O Stoichiometric Combustion Simulator")
st.markdown(
    "This page uses Python Cantera code to simulate stoichiometric methane combustion. Please take note of the following assumptions:\n"
    " - Moles of each species are set using the ideal gas law from the specified T and P.\n"
    " - No heat loss to the surroundings (adiabatic system).\n"
    " - Constant-volume (closed) reactor is used for transient kinetics.\n"
)

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

st.sidebar.header("Simulation Parameters")
st.sidebar.markdown("Adjust the initial conditions to update the simulation in real-time")

# Temperature slider
T_initial = st.sidebar.slider(
    "Initial Temperature (K)",
    min_value=950,
    max_value=2000,
    value=1000,
    step=50,
    help="Temperature range: 500 K to 2000 K (higher temps = faster reaction)"
)

# Pressure slider
p_atm = st.sidebar.slider(
    "Pressure (atm)",
    min_value=0.1,
    max_value=10.0,
    value=1.0,
    step=0.05,
    help="Pressure range: 0.1 to 1.0 atm"
)

# Simulation parameters
st.sidebar.subheader("Simulation Settings")
t_end_ms = st.sidebar.slider(
    "Simulation Time (ms)",
    min_value=1,
    max_value=15,
    value=5,
    step=1
)
t_end = t_end_ms / 1000  # Convert to seconds

# n_steps = st.sidebar.slider(
#     "Number of Time Steps",
#     min_value=100,
#     max_value=1000,
#     value=500,
#     step=100
# )
n_steps = 1000

# ============================================================================
# SIMULATION FUNCTION (No Caching - Fresh calculation each time)
# ============================================================================

def run_simulation(T_init, p_atm, t_end, n_steps):
    """Run the H2 + O2 -> H2O reaction simulation"""
    
    # Convert pressure to Pa
    p = p_atm * ct.one_atm
    
    # Initial composition for stoichiometric CH4 combustion:
    # CH4 + 2 O2 + 7.52 N2 -> CO2 + 2 H2O + 7.52 N2
    initial_composition = 'CH4:1.0, O2:2.0, N2:7.52'
    n_tot = 1.0 + 2.0 + 7.52
    
    # Create solution object
    gas = ct.Solution('gri30.yaml')
    gas.TPX = T_init, p, initial_composition
    
    # Get species indices
    i_ch4 = gas.species_index('CH4')
    i_o2 = gas.species_index('O2')
    i_co2 = gas.species_index('CO2')
    i_h2o = gas.species_index('H2O')
    
    # Create reactor
    reactor = ct.IdealGasReactor(gas)
    reactor.volume = (8.314 * n_tot * T_init)/(p_atm * 101325)  # Volume from ideal gas law (arbitrary)
    
    # Create reactor network
    network = ct.ReactorNet([reactor])
    
    # Time array
    times = np.linspace(0, t_end, n_steps)
    
    # Storage arrays
    concentrations_ch4 = np.zeros(n_steps)
    concentrations_o2 = np.zeros(n_steps)
    concentrations_co2 = np.zeros(n_steps)
    concentrations_h2o = np.zeros(n_steps)
    temperatures = np.zeros(n_steps)
    pressures = np.zeros(n_steps)
    reaction_rates = np.zeros(n_steps)
    mole_fractions_ch4 = np.zeros(n_steps)
    mole_fractions_o2 = np.zeros(n_steps)
    mole_fractions_co2 = np.zeros(n_steps)
    mole_fractions_h2o = np.zeros(n_steps)
    
    # Run simulation
    for i, t in enumerate(times):
        network.advance(t)
        
        concentrations_ch4[i] = reactor.thermo.concentrations[i_ch4]
        concentrations_o2[i] = reactor.thermo.concentrations[i_o2]
        concentrations_co2[i] = reactor.thermo.concentrations[i_co2]
        concentrations_h2o[i] = reactor.thermo.concentrations[i_h2o]
        temperatures[i] = reactor.T
        pressures[i] = reactor.thermo.P
        # Track CO2 formation rate as the main product rate
        reaction_rates[i] = reactor.thermo.creation_rates[i_co2]

        mole_fractions_ch4[i] = reactor.thermo.X[i_ch4]
        mole_fractions_o2[i] = reactor.thermo.X[i_o2]
        mole_fractions_co2[i] = reactor.thermo.X[i_co2]
        mole_fractions_h2o[i] = reactor.thermo.X[i_h2o]
    
    return {
        'times': times,
        'conc_ch4': concentrations_ch4,
        'conc_o2': concentrations_o2,
        'conc_co2': concentrations_co2,
        'conc_h2o': concentrations_h2o,
        'temps': temperatures,
        'pressures': pressures,
        'rates': reaction_rates,
        'x_ch4': mole_fractions_ch4,
        'x_o2': mole_fractions_o2,
        'x_co2': mole_fractions_co2,
        'x_h2o': mole_fractions_h2o,
    }

# ============================================================================
# RUN SIMULATION
# ============================================================================

with st.spinner("Running simulation..."):
    results = run_simulation(T_initial, p_atm, t_end, n_steps)

# ============================================================================
# DISPLAY METRICS
# ============================================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Initial Temperature",
        value=f"{T_initial} K",
        delta=f"{results['temps'][-1] - T_initial:.1f} K change"
    )

with col2:
    st.metric(
        label="Initial Pressure",
        value=f"{p_atm:.2f} atm",
        delta=f"{results['pressures'][-1] / ct.one_atm - p_atm:.3f} atm change"
    )

with col3:
    ch4_consumed = (1 - results['conc_ch4'][-1] / results['conc_ch4'][0]) * 100 if results['conc_ch4'][0] > 0 else 0
    st.metric(
        label="CH₄ Consumed",
        value=f"{ch4_consumed:.2f}%",
        delta=f"{results['conc_ch4'][0] - results['conc_ch4'][-1]:.2e} kmol/m³"
    )

with col4:
    st.metric(
        label="Max CO₂ Rate [kmol/(m³·s)]",
        value=f"{np.max(results['rates']):.2e}"
    )

st.markdown("---")

# ============================================================================
# CREATE PLOTS
# ============================================================================

col1, col2 = st.columns(2)

# Convert times to milliseconds
times_ms = results['times'] * 1000

# ============================================================================
# PLOT 1: Concentration vs Time
# ============================================================================

with col1:
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(times_ms, results['conc_ch4'] * 1e6, 'b-', linewidth=2.5, label='CH₄')
    ax1.plot(times_ms, results['conc_o2'] * 1e6, 'r-', linewidth=2.5, label='O₂')
    ax1.plot(times_ms, results['conc_co2'] * 1e6, 'g-', linewidth=2.5, label='CO₂')
    ax1.set_xlabel('Time [ms]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Concentration [μmol/m³]', fontsize=12, fontweight='bold')
    ax1.set_title('Species Concentrations vs Time', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    st.pyplot(fig1)

# ============================================================================ 
# PLOT 2: Zoomed Concentrations vs Time (region of rapid change)
# ============================================================================

def find_zoom_window(times, conc_arrays, threshold_pct=1.0, pad_frac=0.05, equilib_window=20, slope_tol=100):
    """Find start/end indices where concentrations are actively changing.
    Trims both the front (induction delay) and back (equilibrium tail).
    
    Args:
        times: time array
        conc_arrays: list of concentration 1D arrays
        threshold_pct: percent threshold to detect active change from initial (default 1.0%)
        pad_frac: fraction of total samples to pad before/after active region
        equilib_window: number of samples over which to compute slope for equilibrium detection
        slope_tol: tolerance for slope (absolute change per unit time) to consider stabilized
    
    Returns:
        (t_start, t_end) in same units as times
    """
    conc_stack = np.vstack(conc_arrays)
    c0 = conc_stack[:, 0]
    n = len(times)
    dt = times[1] - times[0] if n > 1 else 1.0  # time step
    
    # Find start: detect where any concentration changes by >= threshold_pct from initial
    denom = np.where(c0 > 0, c0, np.nanmax(conc_stack, axis=1) + 1e-30)
    pct_changes = np.abs((conc_stack - c0[:, None]) / denom[:, None]) * 100.0
    mask_active = np.any(pct_changes >= threshold_pct, axis=0)
    idx_active = np.where(mask_active)[0]
    
    if idx_active.size == 0:
        # fallback: compare to final value
        denom2 = np.where(conc_stack[:, -1] > 0, conc_stack[:, -1], np.nanmax(conc_stack, axis=1) + 1e-30)
        pct2 = np.abs((conc_stack - conc_stack[:, -1][:, None]) / denom2[:, None]) * 100.0
        mask_active = np.any(pct2 >= threshold_pct, axis=0)
        idx_active = np.where(mask_active)[0]
    
    if idx_active.size == 0:
        # no significant change detected: use first 10%
        start_i = 0
        end_i = max(1, int(0.1 * n))
    else:
        # Find end: detect where all concentrations have stabilized (slopes near zero)
        end_i = n - 1
        window_size = min(equilib_window, max(2, n // 10))  # use 10% of data or equilib_window, whichever is smaller
        
        if window_size > 1:
            # Scan from end backwards to find where all species slopes are small
            for i in range(n - window_size, idx_active[-1], -1):
                window = conc_stack[:, i:i+window_size]
                times_window = times[i:i+window_size]
                dt_window = times_window[-1] - times_window[0]
                
                if dt_window > 0:
                    # compute slope (change per unit time) for each species
                    slopes = np.abs(window[:, -1] - window[:, 0]) / dt_window
                    # if all slopes are small, we've reached equilibrium
                    if np.all(slopes < slope_tol):
                        end_i = i
                        break
        
        pad = max(1, int(pad_frac * n))
        start_i = max(0, idx_active[0] - pad)
        end_i = min(n - 1, end_i + pad)
    
    return times[start_i], times[end_i]


with col2:
    # prepare concentration arrays (convert kmol/m^3 -> μmol/m^3 for plotting)
    conc_ch4 = results['conc_ch4'] * 1e6
    conc_o2  = results['conc_o2']  * 1e6
    conc_co2 = results['conc_co2'] * 1e6
    conc_h2o = results['conc_h2o'] * 1e6

    # Find zoom window in seconds using percent-change detection
    t0, t1 = find_zoom_window(
        results['times'],
        [results['conc_ch4'], results['conc_o2'], results['conc_co2'], results['conc_h2o']],
        threshold_pct=10.0,    # change this to tune sensitivity (percent)
        pad_frac=0.05         # fraction of samples to pad before/after region
    )
    # convert to ms for x-axis
    t0_ms = t0 * 1000.0
    t1_ms = t1 * 1000.0

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(times_ms, conc_ch4, 'b-', linewidth=2.5, label='CH₄')
    ax2.plot(times_ms, conc_o2,  'r-', linewidth=2.5, label='O₂')
    ax2.plot(times_ms, conc_co2, 'g-', linewidth=2.5, label='CO₂')
    ax2.plot(times_ms, conc_h2o, 'm-', linewidth=2.0, label='H₂O')

    ax2.set_xlabel('Time [ms]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Concentration [μmol/m³]', fontsize=12, fontweight='bold')
    ax2.set_title('Zoomed Species Concentrations (rapid-change region)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)

    # apply the zoom limits (if t0==t1 the set_xlim will still be valid)
    ax2.set_xlim([t0_ms, t1_ms])

    fig2.tight_layout()
    st.pyplot(fig2)

col3, col4 = st.columns(2)

# ============================================================================
# PLOT 3: H2O Production Rate vs Time
# ============================================================================

with col3:
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(times_ms, results['rates'] * 1e6, 'g-', linewidth=2.5)
    ax3.set_xlabel('Time [ms]', fontsize=12, fontweight='bold')
    ax3.set_ylabel('CO₂ Production Rate [μmol/(m³·s)]', fontsize=12, fontweight='bold')
    ax3.set_title('CO₂ Formation Rate vs Time', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.fill_between(times_ms, results['rates'] * 1e6, alpha=0.3, color='green')
    fig3.tight_layout()
    st.pyplot(fig3)

# ============================================================================
# PLOT 4: Temperature and Pressure vs Time
# ============================================================================

with col4:
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    # Plot temperature on left axis
    color1 = 'tab:orange'
    ax4.set_xlabel('Time [ms]', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Temperature [K]', color=color1, fontsize=12, fontweight='bold')
    ax4.plot(times_ms, results['temps'], color=color1, linewidth=2.5, label='Temperature')
    ax4.tick_params(axis='y', labelcolor=color1)
    ax4.grid(True, alpha=0.3)
    
    # Plot pressure on right axis
    ax4_2 = ax4.twinx()
    color2 = 'tab:purple'
    ax4_2.set_ylabel('Pressure [atm]', color=color2, fontsize=12, fontweight='bold')
    ax4_2.plot(times_ms, results['pressures'] / ct.one_atm, color=color2, linewidth=2.5, 
               linestyle='--', label='Pressure')
    ax4_2.tick_params(axis='y', labelcolor=color2)
    
    ax4.set_title('Temperature & Pressure vs Time', fontsize=13, fontweight='bold')
    
    # Add legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_2.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
    
    fig4.tight_layout()
    st.pyplot(fig4)

st.markdown("---")

# ============================================================================
# DETAILED STATISTICS
# ============================================================================

st.subheader("📊 Detailed Simulation Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Initial Conditions")
    st.write(f"- **Temperature**: {T_initial} K")
    st.write(f"- **Pressure**: {p_atm:.2f} atm")
    st.write(f"- **Simulation Time**: {t_end*1000:.1f} ms")
    st.write(f"- **Time Steps**: {n_steps}")

with col2:
    st.markdown("### Final State")
    st.write(f"- **Final Temperature**: {results['temps'][-1]:.2f} K")
    st.write(f"- **Final Pressure**: {results['pressures'][-1] / ct.one_atm:.4f} atm")
    st.write(f"- **ΔT**: {results['temps'][-1] - T_initial:.2f} K")
    st.write(f"- **ΔP**: {(results['pressures'][-1] - p_atm*ct.one_atm) / ct.one_atm:.4f} atm")

with col3:
    st.markdown("### Reaction Progress")
    ch4_consumed = (1 - results['conc_ch4'][-1] / results['conc_ch4'][0]) * 100 if results['conc_ch4'][0] > 0 else 0
    o2_consumed = (1 - results['conc_o2'][-1] / results['conc_o2'][0]) * 100 if results['conc_o2'][0] > 0 else 0
    st.write(f"- **CH₄ Consumed**: {ch4_consumed:.2f}%")
    st.write(f"- **O₂ Consumed**: {o2_consumed:.2f}%")
    st.write(f"- **Max CO₂ Rate**: {np.max(results['rates']):.2e} kmol/(m³·s)")
    st.write(f"- **Avg CO₂ Rate**: {np.mean(results['rates']):.2e} kmol/(m³·s)")

# ============================================================================
# CONCENTRATION TABLE
# ============================================================================

st.subheader("📋 Concentration Data")

data_table = {
    'Time (ms)': times_ms[::max(1, n_steps//10)],  # Show ~10 rows
    'CH₄ (kmol/m³)': results['conc_ch4'][::max(1, n_steps//10)],
    'O₂ (kmol/m³)': results['conc_o2'][::max(1, n_steps//10)],
    'CO₂ (kmol/m³)': results['conc_co2'][::max(1, n_steps//10)],
    'H₂O (kmol/m³)': results['conc_h2o'][::max(1, n_steps//10)],
    'T (K)': results['temps'][::max(1, n_steps//10)],
    'P (atm)': results['pressures'][::max(1, n_steps//10)] / ct.one_atm,
}

st.dataframe(data_table, width='stretch')

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 12px;">
    <p>Nathan Bryce | MEGN566</p>
    <p>Powered by Cantera | Built with Streamlit and GitHub Copilot</p>
    </div>
    """,
    unsafe_allow_html=True,
)

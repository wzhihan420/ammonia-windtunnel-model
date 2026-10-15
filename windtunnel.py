import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. UI Configuration 
# ==========================================
st.set_page_config(page_title="Ammonia Volatilization Model", layout="wide")
st.title("🌬️ Phase 2: Surface Ammonia Volatilization Kinetics")
st.markdown("Reverse-engineering aerodynamic and chemical parameters under severe **evaporative cooling** conditions (6 m/s wind speed).")

# ==========================================
# 2. Experimental Data Targets 
# ==========================================
# Data extracted from wind tunnel flux measurements
exp_concentrations = np.array([7.0, 14.0, 21.0, 35.0]) # Initial Mass Fraction (%)
exp_fluxes = np.array([0.0845, 0.1184, 0.1748, 0.3524]) # Measured Flux (g/m2.s)

# Measured minimum surface temperatures reflecting latent heat decay
exp_T_surf = np.array([13.50, 10.40, 4.90, -0.10]) # °C

# ==========================================
# 3. Interactive Parameters 
# ==========================================
st.sidebar.header("⚙️ Model Parameters")
st.sidebar.markdown("Adjust global thermodynamic variables to align the model with all discrete targets.")

wind_speed = st.sidebar.number_input(
    "Reference Wind Speed (U10, m/s)", 
    min_value=0.00, max_value=20.00, value=6.00, step=0.10, format="%.2f",
    help="Targeted at 6.0 m/s based on the experimental setup."
)

system_pH = st.sidebar.number_input(
    "Interfacial pH", 
    min_value=7.00, max_value=14.00, value=10.50, step=0.05, format="%.2f",
    help="Global interfacial pH. Must be high enough to counteract the extreme pKa surge at sub-zero temperatures."
)

activity_coeff_base = st.sidebar.number_input(
    "Base Activity Coefficient (γ₀)", 
    min_value=0.05, max_value=2.00, value=0.25, step=0.05, format="%.2f",
    help="Base coefficient. The model applies an exponential multiplier to account for extreme intermolecular repulsion at >20% concentrations."
)

# ==========================================
# 4. Mathematical Core 
# ==========================================
def calculate_theoretical_flux(conc_percent, U10, pH, T_C, gamma_base):
    """
    Computes evaporation flux incorporating transient evaporative cooling and dynamic non-ideality.
    """
    T_K = T_C + 273.15
    
    # Mechanism 1: Chemical Speciation
    pKa = 0.09018 + 2729.92 / T_K
    f_NH3 = 1.0 / (1.0 + 10**(pKa - pH))
    
    # Mechanism 2: Phase Equilibrium 
    K_H = 58.0 * np.exp(4085.0 * (1.0/T_K - 1.0/298.15))
    R_gas = 0.08206 # L*atm/(K*mol)
    H_cc = 1.0 / (K_H * R_gas * T_K) # Dimensionless
    
    # Mechanism 3: Aerodynamic Resistance 
    U_star = 0.03 * U10 
    k_g = 1.0e-3 + 46.2e-3 * U_star * 1.4 # m/s
    
    # Concentration Conversion: % w/w to g/m3
    C_liquid_gm3 = conc_percent * 10000.0 
    
    # Mechanism 4: Dynamic Activity Coefficient Correction
    # Exponentially counteracts the cold temperature penalty at high concentration gradients.
    dynamic_gamma = gamma_base * np.exp(0.06 * conc_percent)
    
    # Interfacial Volatilization Flux
    flux = k_g * H_cc * dynamic_gamma * f_NH3 * C_liquid_gm3
    
    return flux, f_NH3, k_g, H_cc, dynamic_gamma

# ---------------------------------------------------------
# Discrete Case Validation
# ---------------------------------------------------------
model_fluxes = []
for c, T in zip(exp_concentrations, exp_T_surf):
    flux, _, _, _, _ = calculate_theoretical_flux(c, wind_speed, system_pH, T, activity_coeff_base)
    model_fluxes.append(flux)

# Extract current metrics for display (@ 14% as a reference midpoint)
_, current_fNH3, current_kg, current_Hcc, current_gamma = calculate_theoretical_flux(
    14.0, wind_speed, system_pH, exp_T_surf[1], activity_coeff_base
)

# ==========================================
# 5. Visualization Dashboard 
# ==========================================
col1, col2 = st.columns([3, 1])

with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add background bar charts for standard academic comparison
    width = 1.5
    ax.bar(exp_concentrations - width/2, exp_fluxes, width, color="#e74c3c", alpha=0.2, label="_nolegend_")
    ax.bar(exp_concentrations + width/2, model_fluxes, width, color="#2ecc71", alpha=0.2, label="_nolegend_")

    # Plot faint guiding line
    ax.plot(exp_concentrations, model_fluxes, color="#2ecc71", linestyle='--', alpha=0.4, zorder=4)

    # Plot Experimental Targets
    ax.scatter(exp_concentrations, exp_fluxes, color="#e74c3c", marker='o', s=120, zorder=5, 
               edgecolor='black', label="Experimental Targets")
    
    # Plot Model Predictions
    ax.scatter(exp_concentrations, model_fluxes, color="#2ecc71", marker='*', s=300, zorder=6, 
               edgecolor='black', label="Model Predictions (Coupled $T_{surf}$)")
    
    # Annotate specific surface temperatures above the model points
    for i in range(len(exp_concentrations)):
        label_text = f"T = {exp_T_surf[i]:.1f} °C"
        ax.annotate(
            label_text,
            (exp_concentrations[i], model_fluxes[i]),
            textcoords="offset points",
            xytext=(0, 15),
            ha='center',
            fontsize=10,
            fontweight='bold',
            color="#27ae60",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ecc71", lw=1, alpha=0.8)
        )

    # Formatting
    ax.set_title("Discrete Case Validation: Model vs. Experiment @ 6.0 m/s", fontsize=14, fontweight='bold')
    ax.set_xlabel("Initial Mass Fraction of NH3 (%)", fontsize=12)
    ax.set_ylabel("Evaporation Flux (g / m²·s)", fontsize=12)
    ax.set_xticks(exp_concentrations) 
    ax.set_xlim(0, 40)
    ax.set_ylim(0, max(max(exp_fluxes), max(model_fluxes)) * 1.3)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    st.pyplot(fig)

with col2:
    st.subheader("📊 Intermediary Metrics (@ 14%)")
    st.metric("Local Surface Temp", f"{exp_T_surf[1]:.2f} °C")
    st.metric("Dynamic Gamma (γ)", f"{current_gamma:.4f}")
    st.metric("Free NH3 Fraction (f_NH3)", f"{current_fNH3:.2%}")
    st.metric("Dimensionless Henry (Hcc)", f"{current_Hcc:.5f}")
    
    st.info("🎯 **Objective:** Adjust Global pH and Base γ₀. The exponential correction logic ensures non-ideal behavior is captured automatically at 35%.")

# ==========================================
# 6. Theoretical Framework Annotations 
# ==========================================
st.markdown("---")
st.subheader("📚 Theoretical Framework & Governing Equations")

col_eq1, col_eq2 = st.columns(2)

with col_eq1:
    st.markdown("**1. Chemical Speciation (Emerson et al., 1975)**")
    st.latex(r"pK_a = 0.09018 + \frac{2729.92}{T_{surf} + 273.15}")
    st.latex(r"f_{NH3} = \frac{1}{1 + 10^{pK_a - pH_{interfacial}}}")
    
    st.markdown("**2. Aerodynamic Correlation (Mackay & Yeun, 1983)**")
    st.latex(r"k_g = 1.0 \times 10^{-3} + 46.2 \times 10^{-3} \cdot U^* \cdot Sc_G^{-0.67}")

with col_eq2:
    st.markdown("**3. Phase Equilibrium (Sander, 2015)**")
    st.latex(r"H_{cc} = \frac{1}{K_H \cdot R \cdot T_K}")
    
    st.markdown("**4. Interfacial Volatilization Flux**")
    st.latex(r"J = k_g \cdot H_{cc} \cdot (\gamma_0 \cdot e^{k \cdot C_{initial}}) \cdot f_{NH3} \cdot C_{liquid\_total}")

st.caption("Note: The model strictly evaluates discrete targets utilizing isolated transient surface temperatures ($T_{surf}$) derived from evaporative cooling measurements, circumventing erroneous isothermal assumptions. Thermodynamic non-ideality is corrected via an exponential activity coefficient function.")

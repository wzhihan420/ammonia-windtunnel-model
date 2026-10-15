import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. UI Configuration (UI 基础配置)
# ==========================================
st.set_page_config(page_title="Ammonia Volatilization Model", layout="wide")
st.title("🌬️ Phase 2: Wind Tunnel Ammonia Volatilization Kinetics")
st.markdown("Reverse-engineering aerodynamic and chemical parameters to match experimental evaporation fluxes.")

# ==========================================
# 2. Experimental Data Targets (实验靶点数据)
# ==========================================
# Data extracted from "NH3 Wind tunnel.xlsx - flux.csv"
exp_concentrations = np.array([7.0, 14.0, 21.0, 35.0]) # Initial Mass Fraction (%)
exp_fluxes = np.array([0.0845, 0.1184, 0.1748, 0.3524]) # Measured Flux (g/m2.s)

# ==========================================
# 3. Interactive Parameters (交互式参数面板)
# ==========================================
st.sidebar.header("⚙️ Model Parameters")
st.sidebar.markdown("Adjust to match the theoretical curve with experimental scatter points.")

# Missing Parameter 1: Wind Speed
wind_speed = st.sidebar.number_input(
    "Reference Wind Speed (U10, m/s)", 
    min_value=0.00, max_value=20.00, value=2.50, step=0.10, format="%.2f",
    help="Drives the gas-phase mass transfer coefficient (kg)."
)

# Missing Parameter 2: System pH
system_pH = st.sidebar.number_input(
    "Initial System pH", 
    min_value=7.00, max_value=14.00, value=11.20, step=0.05, format="%.2f",
    help="Determines the dissociation fraction of free NH3."
)

# Temperature Parameter (Surface Cooling Effect)
surface_temp = st.sidebar.number_input(
    "Interfacial Temperature (°C)", 
    min_value=0.00, max_value=30.00, value=12.00, step=0.50, format="%.2f",
    help="Surface temp drops due to evaporative latent heat. Influences pKa and Henry's constant."
)

# Activity Coefficient Adjustment (High Concentration Non-ideality)
activity_coeff = st.sidebar.number_input(
    "Activity Coefficient (γ)", 
    min_value=0.10, max_value=2.00, value=0.85, step=0.05, format="%.2f",
    help="Thermodynamic correction for inter-molecular repulsion at >7% concentrations."
)

# ==========================================
# 4. Mathematical Core (理论方程核心引擎)
# ==========================================
def calculate_theoretical_flux(conc_percent, U10, pH, T_C, gamma):
    """
    Computes the theoretical evaporation flux based on Two-Film Theory.
    """
    T_K = T_C + 273.15
    
    # Mechanism 1: Chemical Speciation (Free NH3 Fraction)
    pKa = 0.09018 + 2729.92 / T_K
    f_NH3 = 1.0 / (1.0 + 10**(pKa - pH))
    
    # Mechanism 2: Phase Equilibrium (Henry's Law Constant)
    # KH: mol/(L*atm) approximation for ammonia
    K_H = 58.0 * np.exp(4085.0 * (1.0/T_K - 1.0/298.15))
    R_gas = 0.08206 # L*atm/(K*mol)
    # Dimensionless Henry's constant (Gas/Liquid)
    H_cc = 1.0 / (K_H * R_gas * T_K) 
    
    # Mechanism 3: Aerodynamic Resistance (Mackay & Yeun)
    # Simplified friction velocity (U*) estimation over smooth water
    U_star = 0.03 * U10 
    # Sc_G ~ 0.6 for NH3 in air. (0.6)^-0.67 approx 1.4
    k_g = 1.0e-3 + 46.2e-3 * U_star * 1.4 # m/s
    
    # Concentration Conversion: % w/w to g/m3
    # Assuming solution density approx 1000 kg/m3 for simplicity
    C_liquid_gm3 = conc_percent * 10000.0 
    
    # Flux Integration (Two-Resistance simplified for air C_air = 0)
    # J = k_g * H_cc * gamma * f_NH3 * C_liquid
    flux = k_g * H_cc * gamma * f_NH3 * C_liquid_gm3
    
    return flux, f_NH3, k_g, H_cc

# Generate Theoretical Curve Data
conc_range = np.linspace(0, 40, 100)
flux_curve = [calculate_theoretical_flux(c, wind_speed, system_pH, surface_temp, activity_coeff)[0] for c in conc_range]

# Extract current metrics for display
_, current_fNH3, current_kg, current_Hcc = calculate_theoretical_flux(14.0, wind_speed, system_pH, surface_temp, activity_coeff)

# ==========================================
# 5. Visualization Dashboard (结果可视化看板)
# ==========================================
col1, col2 = st.columns([3, 1])

with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Theoretical Curve
    ax.plot(conc_range, flux_curve, color="#2ecc71", lw=2.5, label="Theoretical Prediction Curve")
    
    # Plot Experimental Scatter Targets
    ax.scatter(exp_concentrations, exp_fluxes, color="#e74c3c", s=100, zorder=5, 
               edgecolor='black', label="Experimental Targets (Wind Tunnel)")
    
    # Formatting
    ax.set_title("Ammonia Evaporation Flux: Model vs. Experimental Validation", fontsize=14)
    ax.set_xlabel("Initial Mass Fraction of NH3 (%)", fontsize=12)
    ax.set_ylabel("Evaporation Flux (g / m²·s)", fontsize=12)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, max(0.4, max(flux_curve) * 1.1))
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    st.pyplot(fig)

with col2:
    st.subheader("📊 Intermediary Metrics")
    st.metric("Free NH3 Fraction (f_NH3)", f"{current_fNH3:.2%}", help="Determined by pH & Temp")
    st.metric("Gas Mass Transfer (kg)", f"{current_kg:.4f} m/s", help="Determined by Wind Speed")
    st.metric("Dimensionless Henry (Hcc)", f"{current_Hcc:.5f}", help="Determined by Temp")
    
    st.info("🎯 **Objective:** Adjust the variables in the sidebar until the green line accurately intersects the red experimental points.")

# ==========================================
# 6. Theoretical Framework Annotations (理论框架说明)
# ==========================================
st.markdown("---")
st.subheader("📚 Theoretical Framework & Governing Equations")

col_eq1, col_eq2 = st.columns(2)

with col_eq1:
    st.markdown("**1. Chemical Speciation (Emerson et al., 1975)**")
    st.latex(r"pK_a = 0.09018 + \frac{2729.92}{T + 273.15}")
    st.latex(r"f_{NH3} = \frac{1}{1 + 10^{pK_a - pH}}")
    
    st.markdown("**2. Aerodynamic Correlation (Mackay & Yeun, 1983)**")
    st.latex(r"k_g = 1.0 \times 10^{-3} + 46.2 \times 10^{-3} \cdot U^* \cdot Sc_G^{-0.67}")

with col_eq2:
    st.markdown("**3. Phase Equilibrium & Henry's Law**")
    st.latex(r"H_{cc} = \frac{1}{K_H \cdot R \cdot T_K}")
    
    st.markdown("**4. Interfacial Volatilization Flux**")
    st.latex(r"J = k_g \cdot H_{cc} \cdot \gamma_{NH3} \cdot f_{NH3} \cdot C_{liquid\_total}")

st.caption("Note: $\gamma_{NH3}$ represents the activity coefficient correcting for thermodynamic non-ideality in highly concentrated solutions.")
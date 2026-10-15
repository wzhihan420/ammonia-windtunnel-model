import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==========================================
# 全局页面配置
# ==========================================
st.set_page_config(page_title="Ammonia Simulation Suite", layout="wide", page_icon="🌊")

# ==========================================
# 侧边栏：全局导航
# ==========================================
st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.radio(
    "Select Simulation Phase:",
    ["Phase 1: Underwater Dissolution", "Phase 2: Surface Volatilization"]
)
st.sidebar.markdown("---")

# =====================================================================
# MODULE 1: PHASE 1 - UNDERWATER DISSOLUTION (来自 app1.py)
# =====================================================================
if app_mode == "Phase 1: Underwater Dissolution":
    st.title("🌊 Phase 1: Ammonia Dissolution Analysis & Theoretical Calculator")
    
    tab1, tab2 = st.tabs(["📊 Module 1: Empirical Model Validation", "🧮 Module 2: First-Principles Calculator"])

    # --- TAB 1: 0D Model Validation ---
    with tab1:
        st.header("📊 0D Coupled Kinetic Model")
        col_in, col_plt = st.columns([1, 2.5])
        
        with col_in:
            st.subheader("⚙️ Calibration Inputs")
            kLa_base = st.number_input("Base kLa (at 14 L/min)", value=0.00016, format="%.5f")
            E_factor = st.number_input("Enhancement Factor (E)", value=5.5, format="%.1f")
            
            st.markdown("---")
            st.subheader("🧪 Environment")
            pH_val = st.slider("System pH", 7.0, 12.0, 10.8)
            temp_val = st.slider("Temperature (°C)", 0.0, 40.0, 20.0)
            
            def get_free_nh3_fraction(pH, temp_C):
                pKa = 0.09018 + 2729.92 / (temp_C + 273.15)
                return 1 / (1 + 10**(pKa - pH))
            
            st.metric("Free NH3 Fraction", f"{get_free_nh3_fraction(pH_val, temp_val):.2%}")

        with col_plt:
            def ammonia_ode(t, y, current_kLa, E, pH, temp):
                TAN = y[0]
                V, C_sat = 20.0, 485000 
                C_bulk_NH3 = TAN * get_free_nh3_fraction(pH, temp)
                if t <= 600:
                    dCdt = (current_kLa * E * (C_sat - C_bulk_NH3)) / V
                else:
                    dCdt = -0.00005 * TAN 
                return [dCdt]

            scenarios = [
                {"flow": 7,  "target": 2600,  "color": "#2ecc71", "label": "7 L/min"},
                {"flow": 14, "target": 5500,  "color": "#3498db", "label": "14 L/min"},
                {"flow": 21, "target": 8400,  "color": "#e67e22", "label": "21 L/min"},
                {"flow": 28, "target": 10881, "color": "#e74c3c", "label": "28 L/min"}
            ]

            fig, ax = plt.subplots(figsize=(10, 6))
            t_eval = np.linspace(0, 720, 240)
            for s in scenarios:
                curr_kLa = kLa_base * (s["flow"] / 14.0)
                sol = solve_ivp(ammonia_ode, [0, 720], [0], args=(curr_kLa, E_factor, pH_val, temp_val), t_eval=t_eval)
                ax.plot(sol.t, sol.y[0], color=s["color"], lw=2.5, label=f'Model: {s["label"]}')
                ax.axhline(y=s["target"], color=s["color"], ls='--', alpha=0.4, label=f'Target: {s["target"]}')

            ax.set_title("Validation: Model vs Experimental Data", fontsize=14)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("TAN Concentration (ppm)")
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    # --- TAB 2: First-Principles Calculator ---
    with tab2:
        st.header("🧮 Theoretical Parameter Estimator")
        st.write("Input physical dimensions to calculate mechanistic coefficients.")

        with st.form("physics_form"):
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                Q_in = st.number_input("Flow Rate Q (L/min)", value=14.0, step=1.0)
                H_in = st.number_input("Water Depth H (m)", value=0.30, step=0.05)
            with col_f2:
                db_in = st.number_input("Bubble Diameter db (m)", value=0.005, format="%.3f")
                vb_in = st.number_input("Rising Velocity vb (m/s)", value=0.25, step=0.05)
            with col_f3:
                kr_in = st.number_input("Reaction Constant kr (1/s)", value=10.0, step=1.0)
                st.write("**Constants:** V=20L, D=1.6e-9 m²/s")
            
            submit_btn = st.form_submit_button("🚀 Calculate Theoretical Factors")

        if submit_btn:
            Q_m3s = Q_in / 60000.0
            V_m3, D_NH3 = 0.02, 1.6e-9
            
            a = (6 * Q_m3s * H_in) / (V_m3 * db_in * vb_in)
            kL = 2 * np.sqrt((D_NH3 * vb_in) / (np.pi * db_in))
            kLa_th = kL * a
            Ha = np.sqrt(kr_in * D_NH3) / kL
            E_th = np.sqrt(1 + Ha**2)

            res_c1, res_c2, res_c3 = st.columns(3)
            res_c1.metric("Interfacial Area (a)", f"{a:.2f} m⁻¹")
            res_c2.metric("Theory kLa", f"{kLa_th:.5f} s⁻¹")
            res_c3.metric("Theory E Factor", f"{E_th:.2f}")
            
            st.success(f"Calculated kLa: {kLa_th:.5f} | Calculated E: {E_th:.1f}. Try these in Module 1!")

        st.markdown("---")
        st.subheader("📚 Theoretical Formulas & Annotations")
        st.write("The mechanistic derivation is based on Higbie's Penetration Theory and Hatta Number analysis:")
        st.latex(r"1. \text{Interfacial Area (a)}: a = \frac{6 \cdot Q_{gas} \cdot H}{V_{water} \cdot d_b \cdot v_b}")
        st.latex(r"2. \text{Liquid-side Coeff (kL)}: k_L = 2 \sqrt{\frac{D_{NH3} \cdot v_b}{\pi \cdot d_b}}")
        st.latex(r"3. \text{Volumetric Coeff (kLa)}: k_L a = k_L \cdot a")
        st.latex(r"4. \text{Hatta Number (Ha)}: Ha = \frac{\sqrt{k_r \cdot D_{NH3}}}{k_L}")
        st.latex(r"5. \text{Enhancement Factor (E)}: E = \sqrt{1 + Ha^2}")


# =====================================================================
# MODULE 2: PHASE 2 - SURFACE VOLATILIZATION (来自 windtunnel_2.py)
# =====================================================================
elif app_mode == "Phase 2: Surface Volatilization":
    st.title("🌬️ Phase 2: Surface Ammonia Volatilization Kinetics")
    st.markdown("Reverse-engineering aerodynamic and chemical parameters under severe **evaporative cooling** conditions (6 m/s wind speed).")

    # Experimental Data
    exp_concentrations = np.array([7.0, 14.0, 21.0, 35.0])
    exp_fluxes = np.array([0.0845, 0.1184, 0.1748, 0.3524])
    exp_T_surf = np.array([13.50, 10.40, 4.90, -0.10])

    # Sidebar Parameters for Phase 2
    st.sidebar.header("⚙️ Phase 2 Parameters")
    wind_speed = st.sidebar.number_input("Reference Wind Speed (U10, m/s)", min_value=0.00, max_value=20.00, value=6.00, step=0.10, format="%.2f")
    system_pH = st.sidebar.number_input("Interfacial pH", min_value=7.00, max_value=14.00, value=10.50, step=0.05, format="%.2f")
    activity_coeff_base = st.sidebar.number_input("Base Activity Coefficient (γ₀)", min_value=0.05, max_value=2.00, value=0.25, step=0.05, format="%.2f")

    def calculate_theoretical_flux(conc_percent, U10, pH, T_C, gamma_base):
        T_K = T_C + 273.15
        pKa = 0.09018 + 2729.92 / T_K
        f_NH3 = 1.0 / (1.0 + 10**(pKa - pH))
        
        K_H = 58.0 * np.exp(4085.0 * (1.0/T_K - 1.0/298.15))
        R_gas = 0.08206
        H_cc = 1.0 / (K_H * R_gas * T_K) 
        
        U_star = 0.03 * U10 
        k_g = 1.0e-3 + 46.2e-3 * U_star * 1.4 
        
        C_liquid_gm3 = conc_percent * 10000.0 
        dynamic_gamma = gamma_base * np.exp(0.06 * conc_percent)
        
        flux = k_g * H_cc * dynamic_gamma * f_NH3 * C_liquid_gm3
        return flux, f_NH3, k_g, H_cc, dynamic_gamma

    model_fluxes = []
    for c, T in zip(exp_concentrations, exp_T_surf):
        flux, _, _, _, _ = calculate_theoretical_flux(c, wind_speed, system_pH, T, activity_coeff_base)
        model_fluxes.append(flux)

    _, current_fNH3, current_kg, current_Hcc, current_gamma = calculate_theoretical_flux(
        14.0, wind_speed, system_pH, exp_T_surf[1], activity_coeff_base
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        width = 1.5
        ax.bar(exp_concentrations - width/2, exp_fluxes, width, color="#e74c3c", alpha=0.2, label="_nolegend_")
        ax.bar(exp_concentrations + width/2, model_fluxes, width, color="#2ecc71", alpha=0.2, label="_nolegend_")
        ax.plot(exp_concentrations, model_fluxes, color="#2ecc71", linestyle='--', alpha=0.4, zorder=4)
        ax.scatter(exp_concentrations, exp_fluxes, color="#e74c3c", marker='o', s=120, zorder=5, edgecolor='black', label="Experimental Targets")
        ax.scatter(exp_concentrations, model_fluxes, color="#2ecc71", marker='*', s=300, zorder=6, edgecolor='black', label="Model Predictions (Coupled $T_{surf}$)")
        
        for i in range(len(exp_concentrations)):
            label_text = f"T = {exp_T_surf[i]:.1f} °C"
            ax.annotate(label_text, (exp_concentrations[i], model_fluxes[i]), textcoords="offset points", xytext=(0, 15), ha='center', fontsize=10, fontweight='bold', color="#27ae60", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ecc71", lw=1, alpha=0.8))

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
        st.info("🎯 **Objective:** Adjust Global pH and Base γ₀ to capture non-ideal behavior automatically at 35%.")

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

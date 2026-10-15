import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from asmt_engine import *

# ==========================================
# 1. 页面基础设置
# ==========================================
st.set_page_config(page_title="ASMT Simulator", page_icon="🌊", layout="wide")
st.title("Ammonia Spill Modeling Tool (ASMT)")
st.markdown("A comprehensive, fully-coupled physics and ecological system computing real-time Eulerian-Lagrangian transport, thermodynamic mass-transfer, and eutrophication ODEs. No empirical shortcuts.")
st.divider()

# ==========================================
# 2. 侧边栏：模块选择与参数定义
# ==========================================
st.sidebar.header("Simulation Mode")
sim_mode = st.sidebar.radio("Select Target Module:", 
["1. Surface Spill Only (Air)", 
"2. Underwater Leak Only (Ocean)", 
"3. Combined Simulation (Both)"])
st.sidebar.divider()

with st.sidebar.form("sim_inputs_form"):
    st.header("1. Spill Characteristics")

    if "Surface" in sim_mode:
        st.caption("Data Source: ASMT Framework (draft.docx)")
        m_total = st.number_input("Total Released Mass (kg)", value=136.0, step=10.0)
        f_vap = st.number_input("Initial Vaporization Fraction", min_value=0.0, max_value=1.0, value=0.45, step=0.05)
        d_depth = st.number_input("Mixing Depth (m)", value=0.0625, format="%.4f")
        depth = 0.0
        duration = 60.0
        
    elif "Underwater" in sim_mode:
        st.caption("Data Source: Wang et al. (2025) Singapore Strait")
        spill_rate = st.number_input("Spill Rate (kg/s)", min_value=0.1, max_value=1800.0, value=82.6, step=1.0)
        duration = st.number_input("Spill Duration (seconds)", min_value=1.0, value=180.0, step=10.0)
        depth = st.number_input("Release Depth (m)", min_value=2.0, max_value=15.0, value=5.0, step=0.5)
        m_total = spill_rate * duration
        d_depth = depth
        f_vap = 0.0 

    else: 
        st.caption("Combined Scenario")
        spill_rate = st.number_input("Spill Rate (kg/s)", min_value=0.1, max_value=1800.0, value=82.6, step=1.0)
        duration = st.number_input("Spill Duration (seconds)", min_value=1.0, value=180.0, step=10.0)
        depth = st.number_input("Release Depth (m)", min_value=2.0, max_value=15.0, value=5.0, step=0.5)
        m_total = spill_rate * duration
        d_depth = depth
        f_vap = 0.0 

    st.header("2. Environmental Conditions")
    default_temp = 8.5 if "Surface" in sim_mode else 28.0
    temp_c = st.number_input("Water Temperature (°C)", min_value=1.0, max_value=35.0, value=default_temp, step=0.5)
    ph = st.number_input("pH Level", min_value=7.0, max_value=11.0, value=11.0 if "Surface" in sim_mode else 8.1, step=0.1)
    salinity = st.number_input("Salinity (PSU)", min_value=0.0, max_value=40.0, value=0.0 if "Surface" in sim_mode else 30.0, step=0.5)
    
    if "Surface" in sim_mode or "Combined" in sim_mode:
        u10 = st.number_input("Wind Speed (m/s) [Air]", min_value=0.1, max_value=25.0, value=4.8, step=0.1)
        fetch = st.number_input("Wind Fetch (m)", value=4.0, step=0.5)
        stability = st.selectbox("Pasquill Stability Class", ['A', 'B', 'C', 'D', 'E', 'F'], index=3)

    if "Underwater" in sim_mode or "Combined" in sim_mode:
        current_u = st.number_input("Sea Current (m/s) [Ocean]", min_value=0.01, max_value=2.0, value=0.30, step=0.05)
        eval_time = st.number_input("Plume Eval Time (Hours)", min_value=0.1, max_value=48.0, value=2.0, step=0.5)
        
    st.subheader("3. Hydrodynamic Dispersion")
    st.caption("Turbulent mixing parameters (Calibrated for Singapore Strait).")
    Dx = st.number_input("Longitudinal Dispersion Dx (m²/s)", min_value=1.0, max_value=500.0, value=125.0, step=5.0)
    Dy = st.number_input("Transverse Dispersion Dy (m²/s)", min_value=1.0, max_value=200.0, value=64.0, step=2.0)

    st.subheader("4. Target Receptor (Validation)")
    st.caption("Coordinate & Time for specific spatial-temporal validation.")
    target_x = st.number_input("Receptor X (Downstream m)", value=378.0, step=10.0)
    target_y = st.number_input("Receptor Y (Cross-stream m)", value=0.0, step=1.0)
    target_t = st.number_input("Receptor T (Time hours)", value=0.35, step=0.05, format="%.2f")

    submitted = st.form_submit_button("▶️ Compute Comprehensive Physics", type="primary", use_container_width=True)

# 专属名词解释面板
with st.sidebar.expander("📖 Glossary of Inputs & Outputs"):
    st.markdown("""
    **Key Inputs:**
    - **Receptor X / Y / T:** The specific coordinate and time mapping to ecological zones (e.g., downstream coral reefs validation).
    - **Plume Eval Time:** The snapshot time for the 2D spatial footprint contour map.
    - **Longitudinal/Transverse Dispersion ($D_x, D_y$):** These define the intensity of turbulent mixing in the water column. Calibrated to reproduce rapid dilution in the Singapore Strait.
    """)

# ==========================================
# 3. 核心计算与结果渲染
# ==========================================
if submitted:
    if "Surface" in sim_mode:
        m_puff, m_dissolved = partition_source(m_total, f_vap)
    else:
        m_puff, m_dissolved = calculate_underwater_rise_dissolution(m_total, depth, temp_c)

    f_unionized = calc_unionized_fraction(temp_c, salinity, ph)
    
    avg_Q = 0.0
    if "Surface" in sim_mode or "Combined" in sim_mode:
        dt, total_time = 1, 3600 
        times = np.arange(0, total_time + dt, dt)
        history_dissolved, history_volatilized, history_bio = [], [], []
        current_m = m_dissolved
        cum_vol, cum_bio = 0.0, 0.0
        k20_sec = 0.005 / 3600 

        for t in times:
            history_dissolved.append(current_m)
            history_volatilized.append(cum_vol)
            history_bio.append(cum_bio)
            
            if current_m > 0:
                rate_vol = calc_volatilization_rate(current_m, f_unionized, u10, fetch, temp_c, d_depth)
                rate_bio = calc_biodegradation_rate(current_m, temp_c, k20=k20_sec)
                dm_vol = rate_vol * dt
                dm_bio = rate_bio * dt
                
                if dm_vol + dm_bio > current_m:
                    dm_vol = current_m * (dm_vol / (dm_vol + dm_bio))
                    dm_bio = current_m * (dm_bio / (dm_vol + dm_bio))
                
                current_m -= (dm_vol + dm_bio)
                cum_vol += dm_vol
                cum_bio += dm_bio
        
        avg_Q = cum_vol / total_time if total_time > 0 else 0

    # ==========================================
    # 结果面板渲染
    # ==========================================
    st.subheader("Phase 1: Thermodynamic Initial Assessment")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mass to Air (Puff/Escaped)", f"{m_puff:.1f} kg")
    col2.metric("Mass to Liquid (Dissolved)", f"{m_dissolved:.1f} kg")
    col3.metric("Un-ionized Fraction (f)", f"{f_unionized*100:.2f} %")
    if "Surface" in sim_mode or "Combined" in sim_mode:
        col4.metric("Avg Emission Rate (Q)", f"{avg_Q*1000:.2f} g/s")
    st.divider()

    fig_col1, fig_col2 = st.columns(2) if "Combined" in sim_mode else (st.container(), st.container())

    # --- 模块 A: 大气扩散展示区 ---
    if "Surface" in sim_mode or "Combined" in sim_mode:
        with fig_col1 if "Combined" in sim_mode else st.container():
            if not "Combined" in sim_mode:
                st.subheader("Fate of Dissolved Ammonia (1 Hour)")
                
                fig_mass = go.Figure()
                fig_mass.add_trace(go.Scatter(x=times/3600, y=history_dissolved, mode='lines', name='Dissolved', line=dict(color='#1f77b4', width=2)))
                fig_mass.add_trace(go.Scatter(x=times/3600, y=history_volatilized, mode='lines', name='Volatilized', line=dict(color='#ff7f0e', width=2)))
                fig_mass.add_trace(go.Scatter(x=times/3600, y=history_bio, mode='lines', name='Biodegraded', line=dict(color='#2ca02c', width=2)))
                fig_mass.update_layout(xaxis_title="Time (hr)", yaxis_title="Mass (kg)", margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_mass, use_container_width=True)
            
            st.subheader("Atmospheric Plume Dispersion (z = 2m)")
            x_grid = np.linspace(-10, 100, 120)
            y_grid = np.linspace(-50, 50, 100)
            X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
            Z_mesh = np.zeros_like(X_mesh)
            for i in range(X_mesh.shape[0]):
                for j in range(X_mesh.shape[1]):
                    Z_mesh[i, j] = gaussian_plume(avg_Q, u10, X_mesh[i, j], Y_mesh[i, j], z=2, H=0, stability_class=stability) * 1e6
            
            fig_air = go.Figure(data=go.Contour(
                z=Z_mesh, x=x_grid, y=y_grid, colorscale='YlOrRd', 
                hovertemplate="Downwind: %{x:.1f} m<br>Crosswind: %{y:.1f} m<br>Conc: %{z:.2f} mg/m³<extra></extra>"
            ))
            fig_air.add_trace(go.Scatter(
                x=[0], y=[0], mode='markers', marker=dict(color='black', symbol='star', size=14, line=dict(color='white', width=1.5)), 
                name='Release Source', cliponaxis=False
            ))
            fig_air.update_layout(
                xaxis_title="Downwind Distance (m)", yaxis_title="Crosswind Distance (m)", 
                xaxis=dict(range=[-5, max(x_grid) + 5]), margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_air, use_container_width=True)

    # --- 模块 B: 水下与生态展示区 ---
    if "Underwater" in sim_mode or "Combined" in sim_mode:
        with fig_col2 if "Combined" in sim_mode else st.container():
            st.subheader("Phase 2: Spatial-Temporal Advection Plume")
            time_sec = eval_time * 3600
            target_t_sec = target_t * 3600
            
            # 1. 定点验证浓度计算
            point_conc_eval = calc_single_point_conc(m_dissolved, depth, current_u, Dx, Dy, target_x, target_y, target_t_sec)
            st.success(f"📍 **Receptor Validation Result:** At downstream **{target_x}m**, cross-stream **{target_y}m**, and specifically at **t = {target_t} hours**, the exact concentration is **{point_conc_eval:.4f} mg/L**.")
            
            # 2. 空间热力图计算 (使用 eval_time)
            X_plume, Y_plume, Z_plume, peak_conc_map = calc_underwater_plume(m_dissolved, depth, current_u, Dx, Dy, time_sec)
            
            fig_ocean = go.Figure(data=go.Contour(
                z=Z_plume, x=X_plume[0], y=Y_plume[:, 0], colorscale='Jet',
                hovertemplate="East: %{x:.1f} m<br>North: %{y:.1f} m<br>Conc: %{z:.2f} mg/L<extra></extra>"
            ))
            fig_ocean.add_trace(go.Scatter(
                x=[0], y=[0], mode='markers', 
                marker=dict(color='red', symbol='star', size=14, line=dict(color='white', width=1.5)), 
                name='Release Point', cliponaxis=False
            ))
            fig_ocean.add_trace(go.Scatter(
                x=[target_x], y=[target_y], mode='markers', 
                marker=dict(color='yellow', symbol='diamond', size=12, line=dict(color='black', width=1)), 
                name=f'Receptor ({target_x}m)', cliponaxis=False
            ))
            
            fig_ocean.update_layout(
                xaxis_title="Distance East / Down-current (m)", yaxis_title="Distance North / Cross-current (m)",
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_ocean, use_container_width=True)

            # 3. 时间穿透曲线
            st.subheader(f"Temporal Breakthrough Curve @ Receptor {target_x}m")
            t_hr_array, conc_time_array = calc_point_concentration_over_time(m_dissolved, depth, current_u, Dx, Dy, target_x, target_y, max_time_hours=target_t*3)
            
            fig_breakthrough = go.Figure()
            fig_breakthrough.add_trace(go.Scatter(
                x=t_hr_array, y=conc_time_array, mode='lines', 
                fill='tozeroy', name='TAN Concentration', line=dict(color='purple', width=2)
            ))
            # 添加致死线参考
            fig_breakthrough.add_hline(y=2.1, line_dash="dash", line_color="red", annotation_text="Fish Lethal Limit (2.1 mg/L)")
            fig_breakthrough.add_vline(x=target_t, line_dash="dash", line_color="red", annotation_text=f"Target Time ({target_t}h)")
            
            fig_breakthrough.update_layout(
                xaxis_title="Time Elapsed (Hours)", yaxis_title="TAN Concentration (mg/L)",
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_breakthrough, use_container_width=True)

            # 4. 生态响应曲线 (ODE)
            st.subheader("Result III: Long-Term Ecological Succession")
            peak_initial_shock = np.max(conc_time_array) if np.max(conc_time_array) > 0 else 14.7 # Worst-case default
            t_days, bio_solution = solve_eutrophication(initial_tan=peak_initial_shock, base_do=6.5, base_php=0.1, temp_c=temp_c, days=30)
            
            fig_eco = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
            fig_eco.add_trace(go.Scatter(x=t_days, y=bio_solution[:, 0], mode='lines', name='TAN (mg/L)', line=dict(color='red', width=2)), row=1, col=1)
            fig_eco.add_trace(go.Scatter(x=t_days, y=bio_solution[:, 1], mode='lines', name='DO (mg/L)', line=dict(color='blue', width=2)), row=1, col=1)
            fig_eco.add_trace(go.Scatter(x=t_days, y=bio_solution[:, 2], mode='lines', name='Phytoplankton (mg/L)', line=dict(color='green', width=2)), row=2, col=1)
            
            fig_eco.add_hline(y=3.0, row=1, col=1, line_dash="dash", line_color="navy", annotation_text="Hypoxia (3.0 mg/L)")
            fig_eco.update_layout(height=600, xaxis2_title="Time Elapsed (Days)", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_eco, use_container_width=True)
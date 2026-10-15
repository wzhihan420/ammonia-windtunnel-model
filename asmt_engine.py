import numpy as np
from scipy.integrate import odeint

# ==========================================
# 模块 A: 表面泄漏物理模块
# ==========================================
def partition_source(M_total, f_vap=0.4):
    M_puff = M_total * f_vap
    M_dissolved = M_total * (1 - f_vap)
    return M_puff, M_dissolved

def calc_unionized_fraction(T_c, S, pH):
    T_k = T_c + 273.15 
    pKa = -0.467 + 0.00113 * S + 2887.9 / T_k 
    f = 1 / (1 + 10**(pKa - pH)) 
    return f

def calc_biodegradation_rate(M, T_c, k20=0.005, theta=1.08):
    k_bio = k20 * (theta ** (T_c - 20)) 
    dm_bio = k_bio * M
    return dm_bio 

def calculate_henry_constant(T_c):
    T_k = T_c + 273.15 
    R = 8.314 
    H_cp_std = 5.9e-4 
    C = 4200 
    H_cp = H_cp_std * np.exp(C * (1/T_k - 1/298.15))
    K_H_dimensionless = 1 / (H_cp * R * T_k)
    return K_H_dimensionless

def calc_volatilization_rate(M, f, U10, X, T_c, d_depth):
    Sc_L = 360 
    Sc_G = 0.6 
    K_H = calculate_henry_constant(T_c)
    u_star = 1e-4 * (333.9 * U10 + 4.356 * (U10**2)) 

    if X < 16:
        k_L = 4.31e-3 * u_star * (Sc_L**-0.5) 
    else:
        k_L = (1.191 + 2.551 * np.log10(X)) * 1e-3 * u_star * (Sc_L**-0.5) 

    nu_air = 1e-6 * (9e-4*(T_c**2) + 0.0911*T_c + 13.28) 
    Re_star = u_star * X / nu_air 
    X_star = X * 9.81 / (u_star**2) 
    k_G = 0.0994 * (Re_star**-0.0894) * (X_star**0.0178) * (Sc_G**(-2/3)) * u_star 

    K = 1 / ((1/k_L) + (1/(K_H * k_G)))
    dm_vol = K * M * f / d_depth 
    return dm_vol 

def get_dispersion_coefficients(x, stability_class='D'):
    x = max(x, 1.0) 
    coeffs = {
        'A': (0.22, 0.0001, -0.5, 0.20, 0.0, 0.0),
        'B': (0.16, 0.0001, -0.5, 0.12, 0.0, 0.0),
        'C': (0.11, 0.0001, -0.5, 0.08, 0.0002, -0.5),
        'D': (0.08, 0.0001, -0.5, 0.06, 0.0015, -0.5), 
        'E': (0.06, 0.0001, -0.5, 0.03, 0.0003, -1.0),
        'F': (0.04, 0.0001, -0.5, 0.016, 0.0003, -1.0)
    }
    ay, by, cy, az, bz, cz = coeffs.get(stability_class, coeffs['D'])
    sigma_y = ay * x * (1 + by * x)**cy
    sigma_z = az * x * (1 + bz * x)**cz
    return sigma_y, sigma_z

def gaussian_plume(Q, U_wind, x, y, z, H=0, stability_class='D'):
    sigma_y, sigma_z = get_dispersion_coefficients(x, stability_class)
    if x <= 0: return 0
    term1 = Q / (2 * np.pi * U_wind * sigma_y * sigma_z) 
    term2 = np.exp(-(y**2) / (2 * sigma_y**2)) 
    term3 = np.exp(-(z - H)**2 / (2 * sigma_z**2)) + np.exp(-(z + H)**2 / (2 * sigma_z**2)) 
    return term1 * term2 * term3

# ==========================================
# 模块 B: 水下泄漏物理模块 (2D 解析)
# ==========================================
def calculate_underwater_rise_dissolution(M_total, depth, T_c):
    # 简化：假设水下释放完全溶解，不考虑气泡上升逸出至空气
    if depth <= 0: return M_total, 0.0
    g, rho_w, rho_g, r_b, C_d, k_L_bubble = 9.81, 1025.0, 0.86, 0.005, 0.44, 4e-4 
    C_sat = 480.0 * np.exp(-0.02 * T_c) 
    U_b = np.sqrt((8 * r_b * g * (rho_w - rho_g)) / (3 * C_d * rho_w))
    rise_time = depth / U_b
    
    dt = 0.1 
    times = np.arange(0, rise_time, dt)
    current_mass = M_total
    for t in times:
        if current_mass <= 0: break
        V_total = current_mass / rho_g
        num_bubbles = V_total / ((4/3) * np.pi * (r_b**3))
        A_total = num_bubbles * 4 * np.pi * (r_b**2)
        dm_dt = k_L_bubble * A_total * C_sat
        dissolved_step = dm_dt * dt
        if dissolved_step > current_mass: dissolved_step = current_mass
        current_mass -= dissolved_step

    vaporized_to_surface = current_mass 
    dissolved_in_water = M_total - vaporized_to_surface
    return vaporized_to_surface, dissolved_in_water

def calc_underwater_plume(dissolved_mass_kg, depth, current_u, Dx, Dy, time_sec):
    """计算二维水下平流扩散羽流 (自适应边界)"""
    t = max(time_sec, 1) 
    center_x = current_u * t
    sigma_x = np.sqrt(2 * Dx * t)
    sigma_y = np.sqrt(2 * Dy * t)
    
    x_min = min(-100, center_x - 3 * sigma_x)
    x_max = max(1000, center_x + 3 * sigma_x)
    y_max = max(500, 3 * sigma_y)
    
    x = np.linspace(x_min, x_max, 200)
    y = np.linspace(-y_max, y_max, 100)
    X, Y = np.meshgrid(x, y)
    
    mass_g = dissolved_mass_kg * 1000
    coefficient = mass_g / (4 * np.pi * t * np.sqrt(Dx * Dy) * depth)
    exponential = np.exp(-((X - current_u * t)**2) / (4 * Dx * t) - (Y**2) / (4 * Dy * t))
    Concentration_matrix = coefficient * exponential
    return X, Y, Concentration_matrix, np.max(Concentration_matrix)

def calc_single_point_conc(dissolved_mass_kg, depth, current_u, Dx, Dy, target_x, target_y, time_sec):
    """计算空间中单一固定点在特定时间的浓度"""
    if time_sec <= 0: return 0.0
    mass_g = dissolved_mass_kg * 1000
    coeff = mass_g / (4 * np.pi * time_sec * np.sqrt(Dx * Dy) * depth)
    exp_term = np.exp(-((target_x - current_u * time_sec)**2) / (4 * Dx * time_sec) - (target_y**2) / (4 * Dy * time_sec))
    return coeff * exp_term

def calc_point_concentration_over_time(dissolved_mass_kg, depth, current_u, Dx, Dy, target_x, target_y, max_time_hours=48):
    """计算特定坐标点随时间演变的浓度数组"""
    t_hr = np.linspace(0.05, max_time_hours, 500) 
    t_sec = t_hr * 3600
    mass_g = dissolved_mass_kg * 1000
    coeff = mass_g / (4 * np.pi * t_sec * np.sqrt(Dx * Dy) * depth)
    exp_term = np.exp(-((target_x - current_u * t_sec)**2) / (4 * Dx * t_sec) - (target_y**2) / (4 * Dy * t_sec))
    conc = coeff * exp_term
    return t_hr, conc

# ==========================================
# 模块 C: 生态富营养化动力学 (ODE)
# ==========================================
def water_quality_odes(y, t, params):
    TAN, DO, Php = y
    k_nit, mu_max, K_N, D_p, k_a, DO_sat, Y_O2_N, Y_O2_Php, Y_Php_N = params

    # 1. 严格边界保护 (浓度不能为负)
    TAN, DO, Php = max(TAN, 0.0), max(DO, 0.0), max(Php, 0.0)

    # 2. 引入 Wang 2025 的 DO 限制项 (Monod Kinetics)
    K_DO = 0.5
    do_limitation = DO / (DO + K_DO) if DO > 0 else 0.0

    # 3. 环境容量限制 (防止藻类爆炸)
    max_php_capacity = 1.5 
    space_limitation = max(0.0, 1.0 - (Php / max_php_capacity))
    G_p = mu_max * (TAN / (K_N + TAN)) * space_limitation

    dPhp_dt = (G_p - D_p) * Php

    # 4. 硝化作用消耗 (受氧气浓度制约)
    nitrification = k_nit * TAN * do_limitation
    dTAN_dt = -nitrification - (G_p * Php / Y_Php_N)

    # 5. DO 消耗计算
    dDO_dt = k_a * (DO_sat - DO) - (Y_O2_N * nitrification) - (Y_O2_Php * D_p * Php)

    # 防止积分器跨步将 DO 跌成负数
    if DO <= 0.01 and dDO_dt < 0:
        dDO_dt = 0.0

    return [dTAN_dt, dDO_dt, dPhp_dt]

def solve_eutrophication(initial_tan, base_do, base_php, temp_c, days=30):
    temp_factor = 1.05 ** (temp_c - 20)
    k_nit = 0.15 * temp_factor
    k_a = 0.2 * temp_factor
    params = (k_nit, 1.2, 0.5, 0.3, k_a, base_do, 4.57, 1.2, 5.0)
    t = np.linspace(0, days, 300) 
    y0 = [initial_tan, base_do, base_php]
    solution = odeint(water_quality_odes, y0, t, args=(params,))
    return t, solution
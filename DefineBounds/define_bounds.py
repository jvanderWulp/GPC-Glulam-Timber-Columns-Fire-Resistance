import numpy as np

def check_uc(
    R_time: float, 
    F_design: float, 
    f_d: float, 
    E_d: float, 
    B: float, 
    H: float, 
    L_eff: float, 
    beta_n: float, 
    d_0: float, 
    beta_c: float,
    e_0: float
):
    """
    Calculates the combined Unity Check (UC_tot = UC_buc + UC_B) for a given fire time exposure (R) according to the reduced cross-section method of Eurocode 5. 

    Parameters:
    ----------
    R_time : float
        Fire exposure time [min].
    F_design : float
        Design axial force in the fire situation [N].
    f_d : float
        Design compressive strength in fire [MPa].
    E_d : float
        Design modulus of elasticity in fire [MPa].
    B : float
        Original cross-section width [mm].
    H : float
        Original cross-section height [mm].
    L_eff : float
        Effective (buckling) length [mm].
    beta_n : float
        Notional charring rate [mm/min].
    d_0 : float
        Zero-strength layer thickness [mm].
    beta_c : float
        Straightness / imperfection factor used in the buckling reduction factor [-].
    e_0 : float
        Initial eccentricity producing a first-order bending moment, M_0 = e_0 * F_design [mm].

    Returns
    -------
    float
        Total unity check UC_tot. 
    """

    
    
    k_0 = R_time / 20.0 if R_time < 20.0 else 1.0
    
    d_char = beta_n * R_time
    d_eff = d_char + (k_0 * d_0) # Eq 4.1
    
    B_eff = B - (2 * d_eff)
    H_eff = H - (2 * d_eff)
    
    if B_eff <= 0.0 or H_eff <= 0.0:
        return float('inf') 
        
    A_eff = B_eff * H_eff
    
    sigma_E = F_design / A_eff
    M_0 = e_0 * F_design
    
    I_eff = 1.0/12.0 * B_eff * H_eff**3 
    
    i_eff = H_eff / np.sqrt(12.0)
    lamda = L_eff / i_eff
    lamda_rel = (lamda / np.pi) * np.sqrt(f_d / E_d)
    
    k_buc = 0.5 * (1 + (beta_c * (lamda_rel - 0.3)) + (lamda_rel**2))
    sqrt_term = np.sqrt(max(0.0, (k_buc**2) - (lamda_rel**2)))
    k_c = 1.0 / (k_buc + sqrt_term)
    
    sigma_R_buc = k_c * f_d
    UC_buc = sigma_E / sigma_R_buc
    
    if I_eff == 0.0 or e_0 == 0.0:
        UC_B = 0.0
    else:
        sigma_M = (M_0 * (H_eff / 2.0)) / I_eff
        f_m_d_fi = f_d 
        UC_B = sigma_M / f_m_d_fi
        
    UC_tot = UC_B + UC_buc
    
    return UC_tot

def inverse_r_resistance(
    L: float,               # Length of column [mm] (Required)
    B: float,               # Column width [mm] (Required, H=B assumed)
    F_COLD: float,          # Applied cold vertical force [N] (Required)
    *,
    e_0: float = 0.0,       # Initial eccentricity [mm] (Default: 0.0)
    eta_fi: float = 0.6,    # Load reduction factor for fire (Default: 0.6)
    k: float = 1.0,         # Buckling length factor (Default: 1.0)
    beta_n: float = 0.7,    # Notional charring rate [mm/min] (Default: 0.7)
    d_0: float = 14.0,       # Zero strength layer [mm] (Default: 14.0)
    k_fi: float = 1.15,     # Strength factor (Default: 1.15)
    E_k: float = 9600.0,    # Characteristic stiffness E_0,05 [N/mm²] (Default: 9600)
    f_k: float = 24.0,      # Characteristic strength f_c,0,k [N/mm²] (Default: 24)
    gamma_m: float = 1.0,   # Material factor in fire (Default: 1.0)
    k_mod_fi: float = 1.0,  # Modification factor (Default: 1.0)
    beta_c: float = 0.1,    # Straightness factor (Default: 0.1)
    tol: float = 0.01,      # Tolerance for R result [min] (Default: 0.01)
    max_r: float = 360.0    # Maximum search time [min] (Default: 360)
):
    """
    Finds the fire resistance time R (in minutes) where UC_tot is closest to 1.0 using the Bisection Method.

    Parameters:
    ----------
    L : float
        Column length [mm].
    B : float
        Column width [mm]. A square cross-section is assumed (H=B). 
    F_COLD : float
        Applied axial force in the ambient design situation [N].
    e_0 : float, optional
        Initial eccentricity [mm]. Default is 0.0.
    eta_fi : float, optional
        Load reduction factor to obtain the fire axial force, Default is 0.6.
    k : float, optional
        Buckling length factor. Default is 1.0.
    beta_n : float, optional
        Notional charring rate [mm/min]. Default is 0.7.
    d_0 : float, optional
        Zero-strength layer thickness [mm]. Default is 14.0.
    k_fi : float, optional
        Fire strength/stiffness factor applied to characteristic properties [-]. Default is 1.15.
    E_k : float, optional
        Characteristic modulus of elasticity at ambient conditions [MPa]. Default is 9600.0.
    f_k : float, optional
        Characteristic compressive strength at ambient conditions [MPa]. Default is 24.0.
    gamma_m : float, optional
        Partial material factor in fire [-]. Default is 1.0.
    k_mod_fi : float, optional
        Modification factor in fire [-]. Default is 1.0.
    beta_c : float, optional
        Straightness / imperfection factor used in the buckling reduction factor [-]. Default is 0.1.
    tol : float, optional
        Absolute tolerance on the bracket width for the bisection search [min]. Default is 0.01.
    max_r : float, optional
        Maximum fire time to consider [min]. Default is 360.0.

    Returns
    -------
    tuple[float, float]
        (R_result, UC_result)
    """
    
    H = B
    
    F_FIRE = F_COLD * eta_fi
    
    f_20 = k_fi * f_k
    E_20 = k_fi * E_k
    f_d = (f_20 / gamma_m) * k_mod_fi
    E_d = (E_20 / gamma_m) * k_mod_fi
    L_eff = L * k

    def uc_func(R):
        return check_uc(
            R, F_FIRE, f_d, E_d, B, H, L_eff, 
            beta_n, d_0, beta_c, e_0
        )

    a = 1.0 
    b = 10.0
    step = 10.0
    
    UC_at_a = uc_func(a)
    
    if UC_at_a > 1.0:
        return a, UC_at_a

    while uc_func(b) < 1.0 and b < max_r:
        b += step
    
    UC_at_b = uc_func(b)
    
    if UC_at_b < 1.0 and b >= max_r:
        return max_r, UC_at_b

    a_R = a
    b_R = b
    
    while (b_R - a_R) > tol:
        R_mid = (a_R + b_R) / 2.0
        UC_mid = uc_func(R_mid)
        
        if UC_mid < 1.0:
            a_R = R_mid
        else:
            b_R = R_mid
            
    R_result = a_R 
    UC_result = uc_func(R_result)
    
    return R_result, UC_result
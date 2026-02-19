import numpy as np
import matplotlib.pyplot as plt
import os

def iso834_T(t_min, T0: float = 20.0):
    """
    ISO 834 standard fire temperature–time curve.

    Parameters
    ----------
    t_min : float | np.ndarray
        Time [min]. May be scalar or NumPy array.
    T0 : float, optional
        Ambient (initial) temperature [°C]. (Default 20 °C)

    Returns
    -------
    np.ndarray
        Fire temperature(s) [°C] at the corresponding time(s).
    """
    return T0 + 345.0 * np.log10(8.0 * t_min + 1.0)

def phi(tstar: np.ndarray):
    """
    Eurocode parametric fire heating shape function phi(t*).

    Parameters
    ----------
    tstar : np.ndarray
        Time variable t* (described by t * tau) in hours, where t is the time in hours and tau is a non-dimensional time-scaling factor.

    Returns
    -------
    np.ndarray
        Gas-temperature rise phi(t*), ranging from 0 at t*=0 to 1 as t* goes to infinity.
    """
    return 1.0 - 0.324*np.exp(-0.2*tstar) - 0.204*np.exp(-1.7*tstar) - 0.472*np.exp(-19.0*tstar)

def invert_phi(y: float, a: float = 0.0, b: float = 30.0, tol: float = 1e-10, itmax: int = 200):
    """
    This function numerically solves phi(t*) = y using the bisection method,
    where phi(t*) is the Eurocode parametric heating function.
    
    The solver automatically expands the search interval such that the upper bound does bracket the root. 

    Parameters
    ----------
    y : float
        Target value of phi(t*). Is between [0, 1]. 
    a : float, optional
        Initial lower bound of the search interval (default = 0.0).
    b : float, optional
        Initial upper bound of the search interval (default = 30.0).
        The interval is automatically expanded if phi(b) < y.
    tol : float, optional
        Absolute convergence tolerance on |phi(t*) − y| (default = 1e−10).
    itmax : int, optional
        Maximum number of iterations (default = 200).

    Returns
    -------
    tstar_h : float
        The non-dimensional time t* such that phi(t*) ≈ y.
    """
    
    y = min(max(y, 1e-12), 1 - 1e-12)
    fa = phi(a) - y
    fb = phi(b) - y
    while fb < 0 and b < 1e6:
        b *= 1.5
        fb = phi(b) - y
    for _ in range(itmax):
        m = 0.5*(a+b)
        fm = phi(m) - y
        if abs(fm) < tol:
            return m
        if fa*fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    print(f"Warning: invert_phi did not converge in {itmax} iterations; returning midpoint.")
    return 0.5*(a+b)

def fit_tau(T0: float, T_max: float, t_h: float):
    """
    Compute the Eurocode time-scaling parameter 'tau' from given fire parameters.
    
    Parameters
    ----------
    T0 : float
        Ambient temperature [°C].
    T_max : float
        Peak gas temperature at time t_h [°C].
    t_h : float
        Time to reach the peak temperature [min].

    Returns
    -------
    tau : float
        Time-scaling parameter.
    """
    
    y = (T_max - T0)/1325.0
    tstar_peak = invert_phi(y)
    t_h_hours = t_h / 60
    return tstar_peak / t_h_hours

def build_fire_curve(T0: float, T_max: float, t_h: float, r_cool: float | None, t_end: float, dt: float, plot_curve: bool = True):
    """
    This functions creates the fire curve:
      - Nonlinear heating (Eurocode parametric fire shape)
      - Linear cooling until ambient
      - Ambient tail to t_end

    Parameters
    ----------
    T0 : float
        Ambient temperature [°C]
    T_max : float
        Peak gas temperature at t_h [°C] (< T0+1325)
    t_h : float
        Heating duration to peak [min]
    r_cool : float | None
        Linear cooling rate [°C/min]; Default is None. When None or <=0 it keeps T_max constant.
    t_end : float
        End time [min]
    dt : float
        Numerical time step for the heating phase [min]
    plot_curve : bool
        If True, show plot.

    Returns
    -------
    t_all : np.ndarray   # time [min]
    T_all : np.ndarray   # gas temperature [°C]
    tau   : float        # fitted time-scaling parameter
    """

    if T_max >= T0 + 1325.0:
        raise ValueError(f"T_max must be < T0+1325 = {T0+1325:.1f} °C (Annex A ceiling).")
    if t_end < t_h:
        raise ValueError("t_end must be ≥ t_h.")

    tau = fit_tau(T0, T_max, t_h)
    t_heat = np.arange(0.0, t_h + dt, dt)
    t_heat_hours = t_heat / 60
    T_heat = T0 + 1325.0 * phi(t_heat_hours * tau)
    T_heat[-1] = T_max

    t_all = t_heat.copy()
    T_all = T_heat.copy()

    if r_cool is None or r_cool <= 0:
        if t_end > t_h:
            t_all = np.r_[t_all, t_end]
            T_all = np.r_[T_all, T_max]
    else:
        t_cool_end = t_h + (T_max - T0) / r_cool
        if t_end <= t_cool_end:
            T_end = T_max - r_cool * (t_end - t_h)
            t_all = np.r_[t_all, t_end]
            T_all = np.r_[T_all, T_end]
        else:
            t_all = np.r_[t_all, t_cool_end]
            T_all = np.r_[T_all, T0]
            if t_end > t_cool_end:
                t_all = np.r_[t_all, t_end]
                T_all = np.r_[T_all, T0]

    if plot_curve:
        plt.figure()
        plt.plot(t_all, T_all, label="Fire curve")
        plt.axvline(t_h, linestyle=":", c="r", linewidth=1, label="t_h")
        plt.xlabel("Time [min]")
        plt.ylabel("Gas temperature [°C]")
        plt.title(
            f"T_max={T_max:.0f} °C at t_h={t_h:.0f} min; "
            f"r_cool={r_cool} °C/min; t_end={t_end:.0f} min"
        )
        plt.grid(True)
        plt.legend()

    return t_all, T_all, tau

def F(tstar: np.ndarray):
    """
    Analytical primitive function (integral) of the Eurocode heating function phi(t*).

    Parameters
    ----------
    tstar : np.ndarray
        Time variable (scalar or array-like) [h]

    Returns
    -------
    np.ndarray
        Value of the integral F(tstar)
    
    """
    return (tstar + (0.324/0.2)*(np.exp(-0.2*tstar) - 1.0) + (0.204/1.7)*(np.exp(-1.7*tstar) - 1.0) + (0.472/19.0)*(np.exp(-19.0*tstar) - 1.0))

def area_firecurve(T0: float, T_max: float, t_h: float, r_cool: float | None, t_end: float, tau: float):
    """
    Exact area under the gas-temperature curve above ambient temperature, up to t_end (°C * min).

    Parameters
    ----------
    T0 : float
        Ambient temperature [°C].
    T_max : float
        Peak gas temperature at time t_h [°C].
    t_h : float
        Heating duration to peak [min].
    r_cool : float | None
        Linear cooling rate [°C/min]. Default is None. When None or <=0 it keeps T_max constant.
    t_end : float
        End time for the integral [min].
    tau : float
        Time-scaling parameter [1/min] such that t* = t * tau.

    Returns
    -------
    H : float
        Area above ambient up to t_end, in (°C * min).
    """
    if t_end <= t_h:
        return (1325.0 * F((t_end / 60) * tau)) * 60

    H_heat = (1325.0 * F((t_h / 60) * tau)) * 60

    delta = T_max - T0
    if r_cool is None or r_cool <= 0:
        # No cooling: hold at T_max until t_end
        return H_heat + delta * (t_end - t_h)

    t_cool_full = delta / r_cool
    t2 = min(t_end, t_h + t_cool_full)
    dt_cool = max(0.0, t2 - t_h)
    H_cool = delta * dt_cool - 0.5 * r_cool * dt_cool**2
    return H_heat + H_cool

def save_fct(t_min: np.ndarray, T_c: np.ndarray, filename: str = "myfire.fct", out_dir: str | None = None):
    """
    Save a SAFIR firecurve .fct file with time in seconds and corresponding temperatures in Celcius. Delimiter is double-space.

    Parameters
    ----------
    t_min : np.ndarray
        Time array in minutes.
    T_c : np.ndarray
        Corresponding temperature array in °C.
    filename : str, optional
        Desired output filename (≤10 characters before .fct). Default is 'myfire.fct'.
    out_dir : str | None, optional
        Output folder path. If None, saves to the current working directory.

    Returns
    -------
    out_path : str
        Full path to the saved .fct file.
    """

    assert t_min.shape == T_c.shape, f'Time and temperature needs to be of same size'

    if out_dir is None:
        out_dir = os.getcwd() 
    os.makedirs(out_dir, exist_ok=True)

    root, ext = os.path.splitext(filename)
    if not ext:
        ext = ".fct"
    if len(root) > 6:
        root = root[0] + root[-5:]
        
    out_path = os.path.join(out_dir, f"{root}{ext}")

    t_sec = np.asarray(t_min, dtype=float) * 60 #np.rint(np.asarray(t_min, dtype=float) * 60.0).astype(int)
    T_vals = np.asarray(T_c, dtype=float)

    _, keep_idx = np.unique(t_sec, return_index=True)
    keep_idx.sort()
    t_sec = t_sec[keep_idx]
    T_vals = T_vals[keep_idx]

    np.savetxt(out_path, np.column_stack([t_sec, T_vals]), fmt=["%.4f", "%.1f"], delimiter="  ")
    print(f"File saved at: {out_path}")
    return out_path

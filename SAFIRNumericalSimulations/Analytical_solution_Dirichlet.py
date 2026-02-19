import numpy as np
import matplotlib.pyplot as plt


def analytical_dirichlet(
    B,
    rho,
    T_max,
    t_h,
    r_c,
    *,
    c: float = 1700,
    k: float = 0.2,
    T0: float = 20.0,
    n_terms: int = 60,
    plot: bool = True,
    xlim: tuple = (0, 0),
):
    """
    Evaluate the simplified analytical solution at the centre of the cross-section.

    Parameters
    ----------
    B : float
        Section width [m].
    rho : float
        Density [kg/m^3].
    T_max : float
        Maximum gas temperature [°C].
    t_h : float
        Duration of the heating phase [min].
    r_c : float
        Cooling rate [°C/min] (positive value).
    c : float, optional
        Specific heat capacity [J/kgK].
    k : float, optional
        Thermal conductivity [W/mK].
    T0 : float, optional
        Ambient temperature [°C].
    n_terms : int, optional
        Number of eigenmodes used in the solution.
    plot : bool, optional
        If True, plot T(L,t) and T_g(t).
    xlim : tuple, optional
        Optional x-axis limits for the plot (in minutes).

    Returns
    -------
    t_peak_min : float
        Time of maximum temperature at the centre [min].
    T_peak : float
        Maximum temperature at the centre [°C].
    """
    L = B / 2.0
    t_h = t_h * 60.0
    r_c = r_c / 60.0

    alpha = k / (rho * c)
    pi = np.pi
    m1 = (T_max - T0) / t_h
    m2 = -r_c
    t_c = t_h + (T_max - T0) / r_c

    n = np.arange(n_terms)
    mu_n = ((n + 0.5) * pi) / L
    lambda_n = alpha * mu_n**2
    b_n = 2.0 / ((n + 0.5) * pi)
    sin_term_L = np.sin(mu_n * L)

    t_max_plot = t_c + 2.0 * (L**2 / alpha)
    t1 = np.linspace(0.0, t_h, 20, endpoint=False)
    t2 = np.linspace(t_h,  t_c, 20, endpoint=False)
    t3 = np.linspace(t_c,  t_max_plot, 1000)   
    t = np.concatenate([t1, t2, t3])

    I = I_n(t, n_terms, t_h, t_c, m1, m2, lambda_n)
    Tg_vec = Tg(t, t_h, t_c, T0, T_max, m1, m2)
    T_L = Tg_vec - np.sum(b_n[:, None] * sin_term_L[:, None] * I, axis=0)

    idx_peak = int(np.argmax(T_L))
    t_peak_min = t[idx_peak] / 60.0
    T_peak = T_L[idx_peak]

    if plot:
        plt.figure(figsize=(8, 4.5))
        plt.plot(t / 60.0, T_L, label="T(L,t)", linewidth=2)
        plt.plot(t / 60.0, Tg_vec, "--", label="T_g(t)", alpha=0.7)
        plt.plot(
            t_peak_min,
            T_peak,
            "o",
            c="r",
            label=f"({t_peak_min:.1f} ; {T_peak:.1f})",
        )
        plt.xlabel("Time [min]")
        plt.ylabel("Temperature [°C]")
        plt.title("Temperature response at centre of simplified analytical solution")
        if any(xlim):
            plt.xlim(xlim[0], xlim[1])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    return t_peak_min, T_peak


def Tg(t, t_h, t_c, T0, T_max, m1, m2):
    """
    Gas temperature T_g(t) for the linear approximated fire curve.

    Parameters
    ----------
    t : array_like
        Time array [s].
    t_h : float
        End of heating phase [s].
    t_c : float
        End of cooling phase [s].
    T0 : float
        Ambient temperature [°C].
    T_max : float
        Maximum gas temperature [°C].
    m1 : float
        Heating rate [°C/s].
    m2 : float
        Cooling rate [°C/s] (negative).

    Returns
    -------
    out : ndarray
        Gas temperature T_g(t) [°C] at each time step.
    """
    if np.isscalar(t):
        t = np.array([t])
    out = np.zeros_like(t)

    mask1 = t <= t_h
    out[mask1] = T0 + m1 * t[mask1]

    mask2 = (t > t_h) & (t <= t_c)
    out[mask2] = T_max + m2 * (t[mask2] - t_h)

    mask3 = t > t_c
    out[mask3] = T0

    return out


def I_n(t, n_terms, t_h, t_c, m1, m2, lambda_n):
    """
    I_n(t) for each eigenmode at x = L.

    Parameters
    ----------
    t : array_like
        Time array [s].
    n_terms : int
        Number of eigenmodes used.
    t_h : float
        End of heating phase [s].
    t_c : float
        End of cooling phase [s].
    m1 : float
        Heating rate [°C/s].
    m2 : float
        Cooling rate [°C/s] (negative).
    lambda_n : ndarray
        Modal decay rates lambda_n = alpha * mu_n^2 [1/s].

    Returns
    -------
    I : ndarray
        Array of shape (n_terms, len(t)) with I_n(t) values.
    """
    I = np.zeros((n_terms, len(t)))
    for i, ti in enumerate(t):
        if ti <= t_h:
            I[:, i] = (m1 / lambda_n) * (1.0 - np.exp(-lambda_n * ti))
        elif ti <= t_c:
            term1 = (m1 / lambda_n) * (
                np.exp(-lambda_n * (ti - t_h)) - np.exp(-lambda_n * ti)
            )
            term2 = (m2 / lambda_n) * (1.0 - np.exp(-lambda_n * (ti - t_h)))
            I[:, i] = term1 + term2
        else:
            term1 = (m1 / lambda_n) * (
                np.exp(-lambda_n * (ti - t_h)) - np.exp(-lambda_n * ti)
            )
            term2 = (m2 / lambda_n) * (
                np.exp(-lambda_n * (ti - t_c)) - np.exp(-lambda_n * (ti - t_h))
            )
            I[:, i] = term1 + term2
    return I

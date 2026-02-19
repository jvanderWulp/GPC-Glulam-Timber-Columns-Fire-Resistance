import re
from pathlib import Path
import numpy as np
import pandas as pd
import os

def char_depth_cumulative_temp(
    t,
    T,
    *,
    time_unit: str = "s",          
    temp_unit: str = "C",          
    constant: float = 1.35e5,
    exponent: float = 1.0/1.6,
    return_rate: bool = True,
    freeze_threshold_C: float = 300.0
):
    """
    Compute charring depth from a fire curve using the draft Eurocode 5 cumulative-temperature model. 
    In addition a physical temperature threshold is implemented when temperature drops below freeze_threshold_C after having reached it before the charring depth is held constant. 
    The calculation of the integral is numerically performed with the trapeziodal rule. 

    Parameters
    ----------
    t : np.ndarray
        Time array. Units set by `time_unit`.
    T : np.ndarray
        Corresponding gas temperatures to time instances. Units set by `temp_unit`.
    time_unit : str, optional
        Unit of the time input. Choice between {"s","min"} (default "s")
        If "min", converted internally to seconds.
    temp_unit : str, optional
        Unit of temperature input. Choice between {"C","K"} (default "C")
        If "K", converted internally to Celcius.
    constant : float, optional
        Calibration constant of cumulative temperature model. 
        Default 1.35e5, corresponding to eurocode cumultive temperature model formula.
    exponent : float, optional
        Exponent of of cumulative temperature model.
        Default 1.0/1.6, corresponding to eurocode cumultive temperature model formula.
    return_rate : bool, optional
        Option to return array of charring rates in m/s (Default True)
    freeze_threshold_C : float, optional
        Threshold in Celsius used for freezing the charring depth, once threshold has been reached during the cooling of the fire. (Default 300.0 degrees celcius)

    Returns
    -------
    d_char : np.ndarray
        Charring depth time history in metres.
    beta : np.ndarray, optional
        Charring rate in m/s.
    """
    t = np.asarray(t, dtype=float)
    T = np.asarray(T, dtype=float)
    if t.shape != T.shape:
        raise ValueError("t and T must have the same shape.")

    if time_unit.lower().startswith("s"):
        t = t / 60.0
    elif time_unit.lower() == "min":
        t = t
    else:
        raise ValueError("time_unit must be 's' or 'min'.")

    if temp_unit.upper() == "C":
        T = T + 273.15
    elif temp_unit.upper() == "K":
        T = T
    else:
        raise ValueError("temp_unit must be 'C' or 'K'.")

    freeze_threshold_K = freeze_threshold_C + 273.15
    
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing.")

    if np.isnan(T).any():
        nans = np.isnan(T)
        T[nans] = np.interp(t[nans], t[~nans], T[~nans])

    T2 = T**2
    I = np.zeros_like(t)
    I[1:] = np.cumsum(0.5 * (T2[:-1] + T2[1:]) * np.diff(t))

    d_char = (I / constant) ** exponent

    freeze_idx = None
    reached_threshold = False
    for i in range(len(T)):
        if T[i] >= freeze_threshold_K:
            reached_threshold = True
        elif reached_threshold and T[i] < freeze_threshold_K:
            freeze_idx = i
            break

    if freeze_idx is not None:
        d_char[freeze_idx:] = d_char[freeze_idx]

    if not return_rate:
        return d_char

    beta = np.zeros_like(d_char)
    dt = np.diff(t)
    beta[1:-1] = (d_char[2:] - d_char[:-2]) / (t[2:] - t[:-2])
    beta[0] = (d_char[1] - d_char[0]) / dt[0]
    beta[-1] = (d_char[-1] - d_char[-2]) / dt[-1]

    if freeze_idx is not None:
        beta[freeze_idx:] = 0.0

    # Convert to m/s
    beta = beta / 60000
    
    return d_char, beta

def make_flux_curve(beta_rates, H_r, rho, *, eff: float = 0.8):
    """
    Calculation of additional flux values corresponding to the charring rates. 

    Parameters
    ----------
    beta_rates : np.ndarray
        Charring rates [m/s]
    H_r : float
        Heat of combustion [J/kg]
    rho : float
        Dry density without moisture [kg/m^3]
    eff : float, optional
        Efficiency of the combustion process. How much heat is given back to the conduction by combustion (Default 0.8)

    Returns
    -------
    q : np.ndarray
        Flux values corresponding to the charring rates [W/m^2]
    """
    q = beta_rates * H_r * rho * eff
    return q

def save_flux(t_min: np.ndarray, flux: np.ndarray, filename: str = "myflux.txt", out_dir: str | None = None):
    """
    Save a SAFIR time heatflux curve .txt file with time in seconds and corresponding heat flux in W/m^2. Delimiter is double-space.

    Parameters
    ----------
    t_min : np.ndarray
        Time array in minutes.
    flux : np.ndarray
        Corresponding heat flux in W/m^2. 
    filename : str, optional
        Desired output filename (≤10 chars before .txt). Default is 'myflux.txt'.
    out_dir : str | None, optional
        Output folder path. If None, saves to the current working directory.

    Returns
    -------
    out_path : str
        Full path to the saved .txt file.
    """

    assert t_min.shape == flux.shape, f'Time and temperature needs to be of same size'

    if out_dir is None:
        out_dir = os.getcwd() 
    os.makedirs(out_dir, exist_ok=True)

    root, ext = os.path.splitext(filename)
    if not ext:
        ext = ".txt"
    if len(root) > 6:
        root = root[0] + root[-5:]
        
    out_path = os.path.join(out_dir, f"{root}{ext}")

    t_sec = np.asarray(t_min, dtype=float) * 60
    T_vals = np.asarray(flux, dtype=float)

    _, keep_idx = np.unique(t_sec, return_index=True)
    keep_idx.sort()
    t_sec = t_sec[keep_idx]
    T_vals = T_vals[keep_idx]

    np.savetxt(out_path, np.column_stack([t_sec, T_vals]), fmt=["%.4f", "%.1f"], delimiter="  ")
    print(f"File saved at: {out_path}")
    return out_path

def interp_time(fine_size: float, B: float, time_s, depths):
    """
    Calculates corresponding times of charring depth in seconds to the positions of the created mesh grid. 
    
    Parameters
    ----------
    fine_size : float
        Fine grid spacing [m] used from 0 to B/4.
    B : float
        Section width [m]. Coarse grid (2*fine_size) is used from B/4 to B/2.
    time_s : array-like
        Times [s], same length/shape as `depths`.
    depths : array-like
        Depths [m], same length/shape as `time_s`.

    Returns
    -------
    times_out : np.ndarray
        Interpolated times [s] corresponding to `positions`.
    positions : np.ndarray
        Positions [m] of the fine/coarse grid.
    """
    t = np.asarray(time_s, dtype=float).ravel()
    x = np.asarray(depths, dtype=float).ravel()
    if t.shape != x.shape:
        raise ValueError("time_s and depths must have the same shape and length.")
    
    end_depth = float(x[-1])
    if end_depth > (B/2):
        raise ValueError("Full cross-section is being charred (end_depth > B/2).")

    coarse_size = 2.0 * fine_size
    positions = [0.0]

    n_fine = int((B/4) / fine_size)
    for i in range(1, n_fine + 1):
        d = i * fine_size
        if d <= end_depth:
            positions.append(d)

    n_coarse = int((B/4) / coarse_size)
    base = B / 4
    for i in range(1, n_coarse + 1):
        d = base + i * coarse_size
        if d <= end_depth:
            positions.append(d)
    
    order = np.argsort(x)
    xs = x[order]
    ts = t[order]
    
    dup_mask = np.diff(xs) == 0
    if not np.any(dup_mask):
        xu, tu = xs, ts
    else:
        ux, first_idx, counts = np.unique(xs, return_index=True, return_counts=True)
        last_idx = first_idx + counts - 1
        xu = ux
        tu = ts[last_idx]
    
    if xu.size < 2:
        raise ValueError("Not enough distinct depth points for interpolation.")
    
    if positions:
        if positions[0] < xu[0] or positions[-1] > xu[-1]:
            raise ValueError("Requested positions fall outside measured depth range; extrapolation is not allowed.")
    
    times_out = np.interp(positions, xu, tu, left=np.nan, right=np.nan)
    if np.isnan(times_out).any():
        raise ValueError("Interpolation produced NaN (would require extrapolation).")

    positions = np.array(positions)
    return times_out, positions

def save_flux_segments(t_min: np.ndarray, flux: np.ndarray, time_instances_sec: np.ndarray, out_dir: str | None = None):
    """
    Split a global time–flux curve into multiple 'flux#.txt' files.

    Each file contains:
      - all original global times,
      - the segment boundaries,
      - the flux linearly interpolated within the segment and set to 0 outside.

    Parameters
    ----------
    t_min : array-like
        Global times in minutes.
    flux : array-like
        Global heat flux values [W/m^2], same shape as t_min.
    time_instances_sec : array-like
        Increasing sequence of time bounds in seconds:
    out_dir : str | None
        Output folder (created if not included).

    Returns
    -------
    list[str]
        Full paths to the written files: flux1.txt, flux2.txt, ...
    """
    t = np.asarray(t_min, dtype=float).ravel()
    q = np.asarray(flux, dtype=float).ravel()
    if t.shape != q.shape:
        raise ValueError("t_min and flux must have the same shape and length.")
    if t.size < 2:
        raise ValueError("Need at least two points in the global time–flux curve.")

    bounds = np.asarray(time_instances_sec, dtype=float).ravel()
    if bounds.size < 1:
        return []
    if not np.all(np.diff(bounds) > 0):
        raise ValueError("time_instances_sec must be strictly increasing.")

    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    t_sec = t * 60.0
    order = np.argsort(t_sec)
    t_sec = t_sec[order]
    q = q[order]
    uniq_t, first_idx = np.unique(t_sec, return_index=True)
    t_sec = uniq_t
    q = q[first_idx]

    if t_sec.size < 2:
        raise ValueError("After removing duplicate times, fewer than two points remain.")

    t_min_global, t_max_global = t_sec[0], t_sec[-1]

    if bounds[0] < t_min_global or bounds[-1] > t_max_global:
        raise ValueError("One or more time boundaries fall outside the global time range; extrapolation is not allowed.")

    extended_bounds = np.concatenate([bounds, [t_max_global]])

    out_paths: list[str] = []

    q_at_global = q 

    for i in range(extended_bounds.size - 1):
        t0 = float(extended_bounds[i])
        t1 = float(extended_bounds[i + 1])

        t_seg = t_sec

        need_t0 = (t0 < t_seg[0]) or (t0 > t_seg[-1]) or (np.searchsorted(t_seg, t0) == t_seg.size or t_seg[np.searchsorted(t_seg, t0)] != t0)
        need_t1 = (t1 < t_seg[0]) or (t1 > t_seg[-1]) or (np.searchsorted(t_seg, t1) == t_seg.size or t_seg[np.searchsorted(t_seg, t1)] != t1)

        if need_t0:
            idx0 = np.searchsorted(t_seg, t0)
            t_seg = np.insert(t_seg, idx0, t0)

        if need_t1:
            idx1 = np.searchsorted(t_seg, t1)
            t_seg = np.insert(t_seg, idx1, t1)

        q_seg = np.interp(t_seg, t_sec, q_at_global)

        outside = (t_seg < t0) | (t_seg > t1)
        q_seg[outside] = 0.0

        fname = f"flux{i+1}.txt"
        out_path = os.path.join(out_dir, fname)
        np.savetxt(out_path,
                   np.column_stack([t_seg, q_seg]),
                   fmt=["%.4f", "%.1f"],
                   delimiter="  ")
        out_paths.append(out_path)

    return out_paths
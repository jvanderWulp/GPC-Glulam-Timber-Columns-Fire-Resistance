from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D


def plot_acquisition_values(global_iters, acq_values, save_path: Path):
    """
    Plot acquisition values vs global iteration number.

    Parameters
    ----------
    global_iters : array-like (N,)
        Global iteration numbers (x-axis).
    acq_values : array-like (N,)
        Acquisition values.
    save_path : Path
        Path to save the figure.
    """
    global_iters = np.asarray(global_iters, dtype=int).reshape(-1)
    acq_values = np.asarray(acq_values, dtype=float).reshape(-1)

    if acq_values.size == 0:
        return

    if global_iters.size != acq_values.size:
        raise ValueError("global_iters and acq_values must have the same length.")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(global_iters, acq_values)
    plt.title("Acquisition values")
    plt.ylabel("Acquisition value [-]")
    plt.xlabel("Iteration [-]")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_relative_acq_change(global_iters, acq_rel, save_path: Path,
                            opt_threshold: float | None = None,
                            conv_threshold: float | None = None):
    """
    Plot relative acquisition change vs global iteration number.

    Parameters
    ----------
    global_iters : array-like (N,)
        Global iteration numbers for acquisition values.
    acq_rel : array-like (N-1,)
        Relative change values.
    save_path : Path
        Path to save the fifure. 
    opt_threshold : float | None
        If provided, plots a dashed horizontal line.
    conv_threshold : float | None
        If provided, plots a dashed horizontal line.
    """
    global_iters = np.asarray(global_iters, dtype=int).reshape(-1)
    acq_rel = np.asarray(acq_rel, dtype=float).reshape(-1)

    if acq_rel.size == 0:
        return

    if global_iters.size != acq_rel.size + 1:
        raise ValueError("Expected len(global_iters) == len(acq_rel) + 1 (since acq_rel is a diff series).")

    x = global_iters[1:]

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(x, acq_rel)

    xmin, xmax = int(x.min()), int(x.max())

    if opt_threshold is not None:
        plt.hlines(opt_threshold, xmin, xmax, color='r', ls="--", label=f"Hyperparameter optimisation threshold: {opt_threshold:g}")
    if conv_threshold is not None:
        plt.hlines(conv_threshold, xmin, xmax, color='b', ls="--", label=f"Convergence threshold: {conv_threshold:g}")

    plt.title("Change in relative acquisition values")
    plt.ylabel("Relative acquisition change [-]")
    plt.xlabel("Iteration [-]")
    plt.grid(alpha=0.3)
    if (opt_threshold is not None) or (conv_threshold is not None):
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_log_likelihood(global_iters, log_likelihood_values, save_path: Path):
    """
    Plot log-likelihood vs global iteration number.

    Parameters
    ----------
    global_iters : array-like (N,)
        Global iteration numbers.
    log_likelihood_values : array-like (N,)
        Log-likelihood values per iteration.
    save_path : Path
        Path to save the figure. 
    """
    global_iters = np.asarray(global_iters, dtype=int).reshape(-1)
    log_likelihood_values = np.asarray(log_likelihood_values, dtype=float).reshape(-1)

    if log_likelihood_values.size == 0:
        return

    if global_iters.size != log_likelihood_values.size:
        raise ValueError("global_iters and log_likelihood_values must have the same length.")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(global_iters, log_likelihood_values)
    plt.title("Log-likelihood")
    plt.xlabel("Iteration [-]")
    plt.ylabel("Log-likelihood [-]")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_misclassification(global_iters, miscl_values, save_path: Path,
                           miscl_threshold: float | None = None):
    """
    Plot misclassification ratio vs global iteration number.

    Parameters
    ----------
    global_iters : array-like (N,)
        Global iteration numbers.
    miscl_values : array-like (N,)
        Misclassification ratios.
    save_path : Path
        Path to save the figure. 
    miscl_threshold : float | None
        If provided, plots a dashed horizontal threshold line.
    """
    global_iters = np.asarray(global_iters, dtype=int).reshape(-1)
    miscl_values = np.asarray(miscl_values, dtype=float).reshape(-1)

    if miscl_values.size == 0:
        return

    if global_iters.size != miscl_values.size:
        raise ValueError("global_iters and miscl_values must have the same length.")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(global_iters, miscl_values)

    if miscl_threshold is not None:
        plt.hlines(
            miscl_threshold,
            int(global_iters.min()),
            int(global_iters.max()),
            color='r',
            ls="--",
            label=f"Misclassification threshold: {miscl_threshold:g}",
        )
        plt.legend()

    plt.title("Misclassification ratio")
    plt.ylabel("Ratio [-]")
    plt.xlabel("Iteration [-]")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    

def plot_gp_decision_boundary_2d(
    model,
    X_norm,
    y,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    grid_res: int = 150,
    x_label: str = "x1",
    y_label: str = "x2",
    title: str = "GPC Decision Boundary",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """
    Plot 2D GPC decision boundary in physcial space.

    Parameters
    ----------
    model : Model
        The GPC model. 
    X_norm : (N, 2) array-like
        Training inputs in normalised coordinates.
    y : (N,) array-like
        Binary labels.
    x_bounds : (x_min, x_max)
        Physical bounds for axis 1.
    y_bounds : (y_min, y_max)
        Physical bounds for axis 2.
    grid_res : int
        Grid resolution.
    x_label, y_label : str
        Axis labels.
    title : str
        Figure title.
    save_path : str | Path | None
        If provided, save the figure.
    show : bool
        If True, show the figure. 
    """
    X_norm = np.asarray(X_norm, dtype=float).reshape(-1, 2)
    y = np.asarray(y, dtype=int).ravel()

    if X_norm.shape[1] != 2:
        raise ValueError("plot_gp_decision_boundary_2d expects X_norm with shape (N, 2).")

    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])

    xx_phys, yy_phys = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res),
    )

    xx_norm = (xx_phys - x_min) / (x_max - x_min)
    yy_norm = (yy_phys - y_min) / (y_max - y_min)
    X_grid = np.column_stack([xx_norm.ravel(), yy_norm.ravel()])

    X_phys = np.empty_like(X_norm)
    X_phys[:, 0] = x_min + X_norm[:, 0] * (x_max - x_min)
    X_phys[:, 1] = y_min + X_norm[:, 1] * (y_max - y_min)

    prob = model.predict_proba(X_grid)[:, 1].reshape(xx_phys.shape)

    fig = plt.figure(figsize=(6.5, 6))

    rwblue = LinearSegmentedColormap.from_list(
        "blue_white_red",
        [(0.0, "blue"), (0.5, "white"), (1.0, "red")]
    )

    levels = np.linspace(0, 1, 21)
    cs = plt.contourf(
        xx_phys, yy_phys, prob,
        levels=levels, cmap=rwblue,
        vmin=0.0, vmax=1.0, alpha=0.9
    )
    cbar = plt.colorbar(cs)
    cbar.set_label("P(y = 1)")

    plt.contour(xx_phys, yy_phys, prob, levels=[0.5], linestyles="--", colors="k")

    pts_1 = plt.scatter(
        X_phys[y == 1, 0], X_phys[y == 1, 1],
        c="red", edgecolor="k", label="y = 1 (Failure)"
    )
    pts_0 = plt.scatter(
        X_phys[y == 0, 0], X_phys[y == 0, 1],
        c="blue", edgecolor="k", label="y = 0 (No Failure)"
    )

    decision_proxy = Line2D([0], [0], color="k", linestyle="--", label="Decision boundary")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.legend(
        handles=[pts_1, pts_0, decision_proxy],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        borderaxespad=0.0,
        ncol=3
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_gp_decision_boundary_3d(
    model,
    X_norm,
    y,
    x1_bounds: tuple[float, float],
    x2_bounds: tuple[float, float],
    x3_bounds: tuple[float, float],
    grid_res: int = 60,
    eps: float = 0.03,
    x1_label: str = "x1",
    x2_label: str = "x2",
    x3_label: str = "x3",
    title: str = "GPC Decision Boundary – 3D",
    save_path: str | Path | None = None,
    show: bool = False,
    elev: float = 30,
    azim: float = 45,
):
    """
    Plot 3D GPC decision boundary in physical space. 

    Parameters
    ----------
    model : Model
        The GPC model. 
    X_norm : (N, 3) array-like
        Training inputs in normalised space.
    y : (N,) array-like
        Binary labels.
    x1_bounds, x2_bounds, x3_bounds : (min, max)
        Physical bounds for each axis.
    grid_res : int
        Number of grid points per dimension for boundary approximation.
    eps : float
        Band around 0.5 used to visualise the decision boundary.
    elev, azim : float
        View angles for the 3D plot.
    save_path : str | Path| None
        If provided, save figure here.
    show : bool
        If True, show the figure. 
    """
    X_norm = np.asarray(X_norm, dtype=float).reshape(-1, 3)
    y = np.asarray(y, dtype=int).ravel()
    if X_norm.shape[1] != 3:
        raise ValueError("plot_gp_decision_boundary_3d expects X_norm with shape (N, 3).")
    if y.shape[0] != X_norm.shape[0]:
        raise ValueError("Length mismatch: y must have same number of rows as X_norm.")

    x1_min, x1_max = map(float, x1_bounds)
    x2_min, x2_max = map(float, x2_bounds)
    x3_min, x3_max = map(float, x3_bounds)

    X_phys = np.empty_like(X_norm)
    X_phys[:, 0] = x1_min + X_norm[:, 0] * (x1_max - x1_min)
    X_phys[:, 1] = x2_min + X_norm[:, 1] * (x2_max - x2_min)
    X_phys[:, 2] = x3_min + X_norm[:, 2] * (x3_max - x3_min)

    grid = np.linspace(0.0, 1.0, int(grid_res))
    g1, g2, g3 = np.meshgrid(grid, grid, grid, indexing="ij")
    pts_norm = np.column_stack([g1.ravel(), g2.ravel(), g3.ravel()])

    p1 = model.predict_proba(pts_norm)[:, 1].ravel()
    mask_bd = np.abs(p1 - 0.5) < float(eps)
    bd_pts_norm = pts_norm[mask_bd]

    bd_pts_phys = None
    if bd_pts_norm.size > 0:
        bd_pts_phys = np.empty_like(bd_pts_norm)
        bd_pts_phys[:, 0] = x1_min + bd_pts_norm[:, 0] * (x1_max - x1_min)
        bd_pts_phys[:, 1] = x2_min + bd_pts_norm[:, 1] * (x2_max - x2_min)
        bd_pts_phys[:, 2] = x3_min + bd_pts_norm[:, 2] * (x3_max - x3_min)

    fig = plt.figure(figsize=(8.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    mask1 = (y == 1)
    sc1 = ax.scatter(
        X_phys[mask1, 0], X_phys[mask1, 1], X_phys[mask1, 2],
        c="red", edgecolor="k", s=35, alpha=0.9, label="y = 1 (Failure)"
    )

    mask0 = (y == 0)
    sc0 = ax.scatter(
        X_phys[mask0, 0], X_phys[mask0, 1], X_phys[mask0, 2],
        c="blue", edgecolor="k", s=35, alpha=0.9, label="y = 0 (No Failure)"
    )

    if bd_pts_phys is not None and bd_pts_phys.size > 0:
        ax.scatter(
            bd_pts_phys[:, 0], bd_pts_phys[:, 1], bd_pts_phys[:, 2],
            s=5, c="k", alpha=0.4, label="GP boundary (p≈0.5)"
        )

    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_zlim(x3_min, x3_max)

    ax.set_xlabel(x1_label)
    ax.set_ylabel(x2_label)
    ax.set_zlabel(x3_label)
    ax.set_title(title)

    ax.view_init(elev=float(elev), azim=float(azim))

    ax.legend(loc="best")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_flips_frac_values(global_iters, flip_frac_values, plot_thresh, save_path: Path):
    """
    Plot boundary change metric vs global iteration number.

    Parameters
    ----------
    global_iters : array-like (N,)
        Global iteration numbers.
    flip_frac_values : array-like (N,)
        Values of fraction of changing classes.
    save_path : Path
        Path to save the figure. 
    """
    global_iters = np.asarray(global_iters, dtype=int).reshape(-1)
    flip_frac_values = np.asarray(flip_frac_values, dtype=float).reshape(-1)

    if flip_frac_values.size == 0:
        return

    if global_iters.size != flip_frac_values.size:
        raise ValueError("global_iters and flip_frac_values must have the same length.")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(global_iters, flip_frac_values)
    plt.hlines(plot_thresh, 1, len(flip_frac_values), color='r', ls='--', label="Threshold hyperparameter optimisation")
    plt.title("Boundary change")
    plt.ylabel("Boundary change [-]")
    plt.xlabel("Iteration [-]")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_boundary_spread(global_iters, boundary_frac_values, plot_thresh, save_path: Path):
    """
    Plot boundary spread metric vs global iteration number.

    Parameters
    ----------
    global_iters : array-like (N,)
        Global iteration numbers.
    boundary_frac_values : array-like (N,)
        Values of fraction of thickness of boundary band.
    save_path : Path
        Path to save the figure. 
    """
    global_iters = np.asarray(global_iters, dtype=int).reshape(-1)
    boundary_frac_values = np.asarray(boundary_frac_values, dtype=float).reshape(-1)

    if boundary_frac_values.size == 0:
        return

    if global_iters.size != boundary_frac_values.size:
        raise ValueError("global_iters and boundary_frac_values must have the same length.")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(global_iters, boundary_frac_values)
    plt.hlines(plot_thresh, 1, len(boundary_frac_values), color='r', ls='--', label="Threshold hyperparameter optimisation")
    plt.title("Boundary spread metric")
    plt.ylabel("Boundary spread [-]")
    plt.xlabel("Iteration [-]")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()



def plot_gpc_slice_2d(
    model,
    design_vars,
    dims,
    fixed_values,
    grid_res: int = 150,
    title: str = "GPC slice",
    show: bool = True,
    save_path: str | Path | None = None,
    return_fig: bool = False,
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
    plot_points: bool = True,
    points_mode: str = "near",     # "near" or "all"
    near_tol_norm: float = 0.03,  
    max_points: int | None = 3000,
    dependent_vars: dict | None = None,
):
    """
    Plot a 2D slice of a d-dimensional GPC model by fixing all other dims.

    """

    name_to_idx = {dv.name: i for i, dv in enumerate(design_vars)}
    d = len(design_vars)

    if isinstance(dims[0], str):
        i = name_to_idx[dims[0]]
        j = name_to_idx[dims[1]]
    else:
        i, j = int(dims[0]), int(dims[1])

    if i == j:
        raise ValueError("dims must contain two different dimensions.")
    if not (0 <= i < d and 0 <= j < d):
        raise ValueError(f"dims out of range. Got {dims} for d={d}.")

    if isinstance(fixed_values, dict):
        x_fixed = np.zeros(d, dtype=float)
        for k, dv in enumerate(design_vars):
            if dv.name in fixed_values:
                x_fixed[k] = dv.norm(float(fixed_values[dv.name]))
            else:
                x_fixed[k] = 0.5
    else:
        x_fixed = np.asarray(fixed_values, dtype=float).reshape(-1)
        if x_fixed.size != d:
            raise ValueError(f"fixed_values must have length d={d}. Got {x_fixed.size}.")
        x_fixed = x_fixed.copy()

    if x_bounds is None:
        x_bounds = (design_vars[i].lower, design_vars[i].upper)
    if y_bounds is None:
        y_bounds = (design_vars[j].lower, design_vars[j].upper)

    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    xx_phys, yy_phys = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res),
    )
    
    x_fixed_phys = np.zeros(d, dtype=float)
    for k, dv in enumerate(design_vars):
        if isinstance(fixed_values, dict) and dv.name in fixed_values:
            x_fixed_phys[k] = float(fixed_values[dv.name])
        else:
            x_fixed_phys[k] = dv.denorm(0.5)
    
    N = xx_phys.size
    X_phys_full = np.tile(x_fixed_phys.reshape(1, -1), (N, 1))
    
    X_phys_full[:, i] = xx_phys.ravel()
    X_phys_full[:, j] = yy_phys.ravel()
    
    if "dependent_vars" in locals() and dependent_vars is not None:
        for dep_name, dep_fn in dependent_vars.items():
            dep_idx = name_to_idx[dep_name]
            X_phys_full[:, dep_idx] = np.asarray(dep_fn(X_phys_full, name_to_idx), dtype=float).ravel()
    

    X_grid = np.empty_like(X_phys_full, dtype=float)
    for k, dv in enumerate(design_vars):
        X_grid[:, k] = dv.norm(X_phys_full[:, k])
    
    X_grid = np.clip(X_grid, 0.0, 1.0)
    
    p1 = model.predict_proba(X_grid)[:, 1].reshape(xx_phys.shape)

    fig, ax = plt.subplots(figsize=(7.0, 6.2))

    rwblue = LinearSegmentedColormap.from_list(
        "blue_white_red",
        [(0.0, "blue"), (0.5, "white"), (1.0, "red")]
    )

    levels = np.linspace(0, 1, 21)
    cs = ax.contourf(
        xx_phys, yy_phys, p1,
        levels=levels,
        cmap=rwblue,
        vmin=0.0, vmax=1.0,
        alpha=0.92
    )
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("P(y = 1)")

    ax.contour(xx_phys, yy_phys, p1, levels=[0.5], linestyles="--", colors="k")

    handles = [Line2D([0], [0], color="k", linestyle="--", label="Decision boundary (p≈0.5)")]

    if plot_points:
        Xtr = np.asarray(model.gpc.X, float)
        ytr = np.asarray(model.gpc.Y).ravel().astype(int)

        if points_mode not in ("near", "all"):
            raise ValueError("points_mode must be 'near' or 'all'.")

        if points_mode == "near":
            mask = np.ones(Xtr.shape[0], dtype=bool)
            for k in range(d):
                if k in (i, j):
                    continue
                mask &= (np.abs(Xtr[:, k] - x_fixed[k]) <= near_tol_norm)
            X_plot = Xtr[mask]
            y_plot = ytr[mask]
        else:
            X_plot = Xtr
            y_plot = ytr

        if max_points is not None and X_plot.shape[0] > max_points:
            idx = np.random.choice(X_plot.shape[0], size=max_points, replace=False)
            X_plot = X_plot[idx]
            y_plot = y_plot[idx]

        x_sc = design_vars[i].denorm(X_plot[:, i])
        y_sc = design_vars[j].denorm(X_plot[:, j])

        pts1 = ax.scatter(x_sc[y_plot == 1], y_sc[y_plot == 1],
                          c="red", edgecolor="k", s=25, label="y = 1 (Failure)")
        pts0 = ax.scatter(x_sc[y_plot == 0], y_sc[y_plot == 0],
                          c="blue", edgecolor="k", s=25, label="y = 0 (No Failure)")

        handles = [pts1, pts0] + handles

        if points_mode == "near":
            ax.text(
                0.02, 0.02,
                f"shown points: {len(X_plot)} (near slice, tol={near_tol_norm:.3f} norm)",
                transform=ax.transAxes,
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
            )

    ax.set_xlabel(f"{design_vars[i].name} [{design_vars[i].unit}]".strip())
    ax.set_ylabel(f"{design_vars[j].name} [{design_vars[j].unit}]".strip())
    ax.set_title(title)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True)

    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=True
    )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_fig:
        return fig, ax
    return None


def plot_gpc_slice_3d(
    model,
    design_vars,
    dims,                     
    fixed_values,            
    grid_res: int = 60,
    eps: float = 0.03,       
    title: str = "GPC slice – 3D",
    show: bool = True,
    save_path: str | Path | None = None,
    return_fig: bool = False, 
    elev: float = 25,
    azim: float = -120,
    dependent_vars: dict | None = None,
    plot_points: bool = True,
    points_mode: str = "near",     # "near" or "all"
    near_tol_norm: float = 0.03,   
    max_points: int | None = 5000,
    x1_bounds: tuple[float, float] | None = None,
    x2_bounds: tuple[float, float] | None = None,
    x3_bounds: tuple[float, float] | None = None,
    boundary_alpha: float = 0.35,
    boundary_size: float = 5.0,
):
    """
    Plot a 3D slice of a d-dimensional GPC by fixing all other dimensions. 

    """

    name_to_idx = {dv.name: i for i, dv in enumerate(design_vars)}
    d = len(design_vars)

    if len(dims) != 3:
        raise ValueError("dims must contain 3 distinct dimensions.")
    if isinstance(dims[0], str):
        i = name_to_idx[dims[0]]
        j = name_to_idx[dims[1]]
        k = name_to_idx[dims[2]]
    else:
        i, j, k = map(int, dims)

    if len({i, j, k}) != 3:
        raise ValueError("dims must be three different dimensions.")
    if not (0 <= i < d and 0 <= j < d and 0 <= k < d):
        raise ValueError(f"dims out of range. Got {dims} for d={d}.")

    if x1_bounds is None:
        x1_bounds = (design_vars[i].lower, design_vars[i].upper)
    if x2_bounds is None:
        x2_bounds = (design_vars[j].lower, design_vars[j].upper)
    if x3_bounds is None:
        x3_bounds = (design_vars[k].lower, design_vars[k].upper)

    x1_min, x1_max = map(float, x1_bounds)
    x2_min, x2_max = map(float, x2_bounds)
    x3_min, x3_max = map(float, x3_bounds)

    if isinstance(fixed_values, dict):
        x_fixed_phys = np.zeros(d, dtype=float)
        for dim, dv in enumerate(design_vars):
            if dv.name in fixed_values:
                x_fixed_phys[dim] = float(fixed_values[dv.name])
            else:
                x_fixed_phys[dim] = dv.denorm(0.5)
        fixed_is_dict = True
    else:
        x_fixed_norm = np.asarray(fixed_values, dtype=float).reshape(-1)
        if x_fixed_norm.size != d:
            raise ValueError(f"fixed_values must have length d={d}. Got {x_fixed_norm.size}.")
        x_fixed_phys = np.array([dv.denorm(x_fixed_norm[dim]) for dim, dv in enumerate(design_vars)], dtype=float)
        fixed_is_dict = False

    g1_phys = np.linspace(x1_min, x1_max, int(grid_res))
    g2_phys = np.linspace(x2_min, x2_max, int(grid_res))
    g3_phys = np.linspace(x3_min, x3_max, int(grid_res))

    G1, G2, G3 = np.meshgrid(g1_phys, g2_phys, g3_phys, indexing="ij")
    N = G1.size

    X_phys_full = np.tile(x_fixed_phys.reshape(1, -1), (N, 1))
    X_phys_full[:, i] = G1.ravel()
    X_phys_full[:, j] = G2.ravel()
    X_phys_full[:, k] = G3.ravel()

    if dependent_vars is not None:
        for dep_name, dep_fn in dependent_vars.items():
            dep_idx = name_to_idx[dep_name]
            X_phys_full[:, dep_idx] = np.asarray(dep_fn(X_phys_full, name_to_idx), dtype=float).ravel()

    X_grid = np.empty_like(X_phys_full, dtype=float)
    for dim, dv in enumerate(design_vars):
        X_grid[:, dim] = dv.norm(X_phys_full[:, dim])

    X_grid = np.clip(X_grid, 0.0, 1.0)

    p1 = model.predict_proba(X_grid)[:, 1].ravel()
    mask_bd = np.abs(p1 - 0.5) < float(eps)

    bd_phys = None
    if np.any(mask_bd):
        bd_phys = np.column_stack([
            X_phys_full[:, i][mask_bd],
            X_phys_full[:, j][mask_bd],
            X_phys_full[:, k][mask_bd],
        ])

    fig = plt.figure(figsize=(9.0, 7.8))
    ax = fig.add_subplot(111, projection="3d")
    handles = []

    if plot_points:
        Xtr = np.asarray(model.gpc.X, float)  
        ytr = np.asarray(model.gpc.Y).ravel().astype(int)

        if points_mode not in ("near", "all"):
            raise ValueError("points_mode must be 'near' or 'all'.")

        if points_mode == "near":
            x_fixed_norm_for_mask = np.array([dv.norm(x_fixed_phys[dim]) for dim, dv in enumerate(design_vars)], dtype=float)

            mask = np.ones(Xtr.shape[0], dtype=bool)
            for dim in range(d):
                if dim in (i, j, k):
                    continue
                mask &= (np.abs(Xtr[:, dim] - x_fixed_norm_for_mask[dim]) <= near_tol_norm)

            Xp = Xtr[mask]
            yp = ytr[mask]
        else:
            Xp = Xtr
            yp = ytr

        if max_points is not None and Xp.shape[0] > max_points:
            idx = np.random.choice(Xp.shape[0], size=max_points, replace=False)
            Xp = Xp[idx]
            yp = yp[idx]

        x_sc = design_vars[i].denorm(Xp[:, i])
        y_sc = design_vars[j].denorm(Xp[:, j])
        z_sc = design_vars[k].denorm(Xp[:, k])

        m1 = (yp == 1)
        m0 = (yp == 0)

        sc1 = ax.scatter(x_sc[m1], y_sc[m1], z_sc[m1],
                         c="red", edgecolor="k", s=35, alpha=0.9, label="y = 1 (Failure)")
        sc0 = ax.scatter(x_sc[m0], y_sc[m0], z_sc[m0],
                         c="blue", edgecolor="k", s=35, alpha=0.9, label="y = 0 (No Failure)")
        handles += [sc1, sc0]

    if bd_phys is not None and bd_phys.size > 0:
        bd = ax.scatter(
            bd_phys[:, 0], bd_phys[:, 1], bd_phys[:, 2],
            s=float(boundary_size), c="k", alpha=float(boundary_alpha),
            label="GP boundary (p≈0.5)"
        )
        handles.append(bd)

    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_zlim(x3_min, x3_max)

    ax.set_xlabel(f"{design_vars[i].name} [{design_vars[i].unit}]".strip())
    ax.set_ylabel(f"{design_vars[j].name} [{design_vars[j].unit}]".strip())
    ax.set_zlabel(f"{design_vars[k].name} [{design_vars[k].unit}]".strip())
    ax.set_title(title)

    ax.view_init(elev=float(elev), azim=float(azim))

    if len(handles) > 0:
        ax.legend(loc="upper center")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_fig:
        return fig, ax
    return None



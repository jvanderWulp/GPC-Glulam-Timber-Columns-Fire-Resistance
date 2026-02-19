from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_gp_decision_boundary(
    model,
    X, y,
    grid_res: int = 100,
    x_bounds: tuple | None = None,
    y_bounds: tuple | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str = "Gaussian Process Classifier – Decision Boundary",
    save_path: str | Path | None = None,
    show: bool = True,
):
    """
    Plot GPC decision boundary.

    Parameters
    ----------
    model : Model
        The GPC model.
    X : array-like, (n_samples, 2)
        Training inputs in normalised space.
    y : array-like, (n_samples,)
        Binary labels.
    grid_res : int
        Resolution of plotting grid.
    x_bounds, y_bounds : tuple | None
        If None: plot in normalised coordinates [0,1] x [0,1].
        If given: physical bounds (x_min, x_max), (y_min, y_max) for axes.
    x_label, y_label : str | None
        Axis labels.
    title : str
        Plot title.
    save_path : str | Path | None
        Path to save the figure. 
    show : bool
        If True, display the figure. If False, do not show it.
    """
    physical_axes = (x_bounds is not None) and (y_bounds is not None)

    if physical_axes:
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds

        xx_plot, yy_plot = np.meshgrid(
            np.linspace(x_min, x_max, grid_res),
            np.linspace(y_min, y_max, grid_res)
        )

        xx_norm = min_max_normalise(xx_plot, x_min, x_max)
        yy_norm = min_max_normalise(yy_plot, y_min, y_max)
        X_grid = np.column_stack([xx_norm.ravel(), yy_norm.ravel()])

        X_plot = np.empty_like(X)
        X_plot[:, 0] = min_max_denormalise(X[:, 0], x_min, x_max)
        X_plot[:, 1] = min_max_denormalise(X[:, 1], y_min, y_max)

    else:
        xx_plot, yy_plot = np.meshgrid(
            np.linspace(0, 1, grid_res),
            np.linspace(0, 1, grid_res)
        )
        X_grid = np.column_stack([xx_plot.ravel(), yy_plot.ravel()])
        X_plot = X  

  
    prob = model.predict_proba(X_grid)[:, 1]
    prob = prob.reshape(xx_plot.shape)

    fig = plt.figure(figsize=(6.5, 6))

    rwblue = LinearSegmentedColormap.from_list(
        "blue_white_red",
        [(0.0, "blue"),
         (0.5, "white"),
         (1.0, "red")]
    )

    levels = np.linspace(0, 1, 21)
    cs = plt.contourf(
        xx_plot, yy_plot, prob,
        levels=levels,
        cmap=rwblue,
        vmin=0.0,
        vmax=1.0,
        alpha=0.9
    )
    cbar = plt.colorbar(cs)
    cbar.set_label("P(y = 1)")

    plt.contour(xx_plot, yy_plot, prob, levels=[0.5], linestyles="--", colors="k")

    pts_1 = plt.scatter(X_plot[y == 1, 0], X_plot[y == 1, 1],
                        c="red", edgecolor="k", label="y = 1 (Failure)")
    pts_0 = plt.scatter(X_plot[y == 0, 0], X_plot[y == 0, 1],
                        c="blue",  edgecolor="k", label="y = 0 (No Failure)")

    decision_proxy = Line2D([0], [0], color="k", linestyle="--",
                            label="Decision boundary")

    if physical_axes:
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
    else:
        plt.xlim(0, 1)
        plt.ylim(0, 1)

    if x_label is None:
        x_label = "x1 (normalised)" if not physical_axes else "x1"
    if y_label is None:
        y_label = "x2 (normalised)" if not physical_axes else "x2"

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.legend(
        handles=[pts_1, pts_0, decision_proxy],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        borderaxespad=0.,
        ncol=3
    )

    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_gp_entropy_field(
    model,
    X_train, y_train,
    grid_res: int = 100,
    x_bounds: tuple | None = None,
    y_bounds: tuple | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str = "GPC predictive entropy over design space",
    save_path: str | Path | None = None,
    show: bool = True,
):
    """
    Plot the predictive entropy for a fitted GPC. 

    Parameters
    ----------
    model : Model
        The GPC model.
    X_train : array-like, (n_samples, 2)
        Training inputs in normalised space. 
    y_train : array-like, (n_samples,)
        Binary labels.
    grid_res : int
        Resolution of plotting grid.
    x_bounds, y_bounds : tuple | None
        If None: plot in normalised coordinates [0,1] x [0,1].
        If given: physical bounds (x_min, x_max), (y_min, y_max) for axes.
    x_label, y_label : str | None
        Axis labels.
    title : str
        Plot title.
    save_path : str | Path | None
        Path to save the figure. 
    show : bool
        If True, display the figure. If False, do not show it.
    """
    
    physical_axes = (x_bounds is not None) and (y_bounds is not None)

    if physical_axes:
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds

        xx_plot, yy_plot = np.meshgrid(
            np.linspace(x_min, x_max, grid_res),
            np.linspace(y_min, y_max, grid_res)
        )

        xx_norm = min_max_normalise(xx_plot, x_min, x_max)
        yy_norm = min_max_normalise(yy_plot, y_min, y_max)
        X_grid = np.column_stack([xx_norm.ravel(), yy_norm.ravel()])

        X_plot = np.empty_like(X_train)
        X_plot[:, 0] = min_max_denormalise(X_train[:, 0], x_min, x_max)
        X_plot[:, 1] = min_max_denormalise(X_train[:, 1], y_min, y_max)

    else:
        xx_plot, yy_plot = np.meshgrid(
            np.linspace(0, 1, grid_res),
            np.linspace(0, 1, grid_res)
        )
        X_grid = np.column_stack([xx_plot.ravel(), yy_plot.ravel()])
        X_plot = X_train  # already normalised

    prob = model.predict_proba(X_grid)[:, 1]

    eps = 1e-12
    p = np.clip(prob, eps, 1 - eps)
    H = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    H = H.reshape(xx_plot.shape)

    fig = plt.figure(figsize=(6.5, 6))

    cs = plt.contourf(xx_plot, yy_plot, H, levels=20)
    cbar = plt.colorbar(cs)
    cbar.set_label("Predictive entropy H(x)")

    pts_1 = plt.scatter(
        X_plot[y_train == 1, 0], X_plot[y_train == 1, 1],
        edgecolor="k", c="red", label="y = 1 (Failure)"
    )
    pts_0 = plt.scatter(
        X_plot[y_train == 0, 0], X_plot[y_train == 0, 1],
        edgecolor="k", c="blue", label="y = 0 (No Failure)"
    )

    if physical_axes:
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
    else:
        plt.xlim(0, 1)
        plt.ylim(0, 1)

    if x_label is None:
        x_label = "x1 (normalised)" if not physical_axes else "x1"
    if y_label is None:
        y_label = "x2 (normalised)" if not physical_axes else "x2"

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.legend(
        handles=[pts_1, pts_0],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        borderaxespad=0.,
        ncol=2
    )

    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_acquisition_history(acq_history: pd.DataFrame, save_path: Path):
    if acq_history.empty:
        return

    acq_sorted = acq_history.sort_values("iteration")

    plt.figure(figsize=(6, 4))
    plt.plot(
        acq_sorted["iteration"],
        acq_sorted["acq"],
        marker="o",
        linestyle="-"
    )
    plt.xlabel("Iterations")
    plt.ylabel("Acquisition value")
    plt.title("Acquisition value vs iterations")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_acquisition_mc_field(
    xspace,
    utilitymat,
    x_star=None,
    x_star_sgd=None,
    X_history=None,
    title="MC acquisition field",
    save_path: str | Path | None = None,
    show: bool = True,
):
    """
    Plot the acquisition values at MCSelector candidates. Can be used together with the MCSelector_2 and Multi_start_SGD 
    when return field is set to True. 

    Parameters
    ----------
    xspace : ndarray, shape (N, 2)
        Candidate points sampled by MCSelector (normalised)
    utilitymat : ndarray, shape (N,)
        Acquisition values at each candidate.
    x_star : ndarray or None
        The chosen x* (best candidate).
    X_history : ndarray | None
        All previously chosen design points, shape (M, 2).
    title : str
        Plot title.
    save_path : str | Path | None
        Path to save the figure. 
    show : bool
        If True, display the figure.
    """
    xspace = np.asarray(xspace, float)
    utilitymat = np.asarray(utilitymat, float)

=    mask = np.isfinite(utilitymat)
    xs = xspace[mask, 0]
    ys = xspace[mask, 1]
    zs = utilitymat[mask]

    fig = plt.figure(figsize=(6.5, 6))

=    cs = plt.tricontourf(xs, ys, zs, levels=20)
    cbar = plt.colorbar(cs)
    cbar.set_label("Acquisition value")

=    plt.scatter(
        xs, ys,
        c="k",
        s=8,
        alpha=0.4,
        label="MC candidates",
    )

=    if X_history is not None and len(X_history) > 0:
        X_history = np.asarray(X_history, float)
        plt.scatter(
            X_history[:, 0],
            X_history[:, 1],
            c="white",
            edgecolor="k",
            s=40,
            label="Previous x*",
        )

=    if x_star is not None:
        x_star = np.asarray(x_star, float).ravel()
        plt.scatter(
            [x_star[0]],
            [x_star[1]],
            c="yellow",
            edgecolor="k",
            s=180,
            marker="*",
            label="Chosen x*",
        )
        
    if x_star_sgd is not None:
        x_star_sgd = np.asarray(x_star_sgd, float).ravel()
        plt.scatter(
            [x_star_sgd[0]],
            [x_star_sgd[1]],
            c="red",
            edgecolor="k",
            s=90,
            marker="*",
            label="Chosen x SGD*",
        )

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("x1 (normalised)")
    plt.ylabel("x2 (normalised)")
    plt.title(title)
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_gp_latent_mean_field(
    model,
    X, y,
    grid_res: int = 100,
    x_bounds: tuple | None = None,
    y_bounds: tuple | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str = r"Latent posterior mean $\mu_f(x)$",
    save_path: str | Path | None = None,
    show: bool = True,
):

    X = np.asarray(X, dtype=float).reshape(-1, 2)
    y = np.asarray(y).ravel().astype(int)

    physical_axes = (x_bounds is not None) and (y_bounds is not None)

    if physical_axes:
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds

        xx_plot, yy_plot = np.meshgrid(
            np.linspace(x_min, x_max, grid_res),
            np.linspace(y_min, y_max, grid_res),
        )

        xx_norm = min_max_normalise(xx_plot, x_min, x_max)
        yy_norm = min_max_normalise(yy_plot, y_min, y_max)
        X_grid = np.column_stack([xx_norm.ravel(), yy_norm.ravel()])

        X_plot = np.empty_like(X)
        X_plot[:, 0] = min_max_denormalise(X[:, 0], x_min, x_max)
        X_plot[:, 1] = min_max_denormalise(X[:, 1], y_min, y_max)

    else:
        xx_plot, yy_plot = np.meshgrid(
            np.linspace(0, 1, grid_res),
            np.linspace(0, 1, grid_res),
        )
        X_grid = np.column_stack([xx_plot.ravel(), yy_plot.ravel()])
        X_plot = X

    mu, _ = model.predict_latent_moments(X_grid)
    mu = mu.reshape(xx_plot.shape)

    fig = plt.figure(figsize=(6.5, 6))
    cs = plt.contourf(xx_plot, yy_plot, mu, levels=25)
    cbar = plt.colorbar(cs)
    cbar.set_label(r"$\mu_f(x)$")

    plt.contour(xx_plot, yy_plot, mu, levels=[0.0], linestyles="--", colors="k")

    pts_1 = plt.scatter(
        X_plot[y == 1, 0], X_plot[y == 1, 1],
        c="red", edgecolor="k", label="y = 1 (Failure)"
    )
    pts_0 = plt.scatter(
        X_plot[y == 0, 0], X_plot[y == 0, 1],
        c="blue", edgecolor="k", label="y = 0 (No Failure)"
    )

    if physical_axes:
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
    else:
        plt.xlim(0, 1)
        plt.ylim(0, 1)

    if x_label is None:
        x_label = "x1 (normalised)" if not physical_axes else "x1"
    if y_label is None:
        y_label = "x2 (normalised)" if not physical_axes else "x2"

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.legend(
        handles=[pts_1, pts_0],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        borderaxespad=0.0,
        ncol=2,
    )

    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_gp_latent_variance_field(
    model,
    X, y,
    grid_res: int = 100,
    x_bounds: tuple | None = None,
    y_bounds: tuple | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str = r"Latent posterior variance $\sigma_f^2(x)$",
    save_path: str | Path | None = None,
    show: bool = True,
):
   
    X = np.asarray(X, dtype=float).reshape(-1, 2)
    y = np.asarray(y).ravel().astype(int)

    physical_axes = (x_bounds is not None) and (y_bounds is not None)

    if physical_axes:
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds

        xx_plot, yy_plot = np.meshgrid(
            np.linspace(x_min, x_max, grid_res),
            np.linspace(y_min, y_max, grid_res),
        )

        xx_norm = min_max_normalise(xx_plot, x_min, x_max)
        yy_norm = min_max_normalise(yy_plot, y_min, y_max)
        X_grid = np.column_stack([xx_norm.ravel(), yy_norm.ravel()])

        X_plot = np.empty_like(X)
        X_plot[:, 0] = min_max_denormalise(X[:, 0], x_min, x_max)
        X_plot[:, 1] = min_max_denormalise(X[:, 1], y_min, y_max)

    else:
        xx_plot, yy_plot = np.meshgrid(
            np.linspace(0, 1, grid_res),
            np.linspace(0, 1, grid_res),
        )
        X_grid = np.column_stack([xx_plot.ravel(), yy_plot.ravel()])
        X_plot = X

    _, var = model.predict_latent_moments(X_grid)
    var = var.reshape(xx_plot.shape)

    fig = plt.figure(figsize=(6.5, 6))
    cs = plt.contourf(xx_plot, yy_plot, var, levels=25)
    cbar = plt.colorbar(cs)
    cbar.set_label(r"$\sigma_f^2(x)$")

    pts_1 = plt.scatter(
        X_plot[y == 1, 0], X_plot[y == 1, 1],
        c="red", edgecolor="k", label="y = 1 (Failure)"
    )
    pts_0 = plt.scatter(
        X_plot[y == 0, 0], X_plot[y == 0, 1],
        c="blue", edgecolor="k", label="y = 0 (No Failure)"
    )

    if physical_axes:
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
    else:
        plt.xlim(0, 1)
        plt.ylim(0, 1)

    if x_label is None:
        x_label = "x1 (normalised)" if not physical_axes else "x1"
    if y_label is None:
        y_label = "x2 (normalised)" if not physical_axes else "x2"

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.legend(
        handles=[pts_1, pts_0],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        borderaxespad=0.0,
        ncol=2,
    )

    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
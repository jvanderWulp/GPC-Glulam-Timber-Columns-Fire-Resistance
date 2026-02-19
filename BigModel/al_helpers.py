from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import GPy


@dataclass
class DesignVar:
    name: str
    lower: float
    upper: float
    unit: str = ""

    def denorm(self, x_norm: float) -> float:
        return self.lower + x_norm * (self.upper - self.lower)

    def norm(self, x_phys: float) -> float:
        return (x_phys - self.lower) / (self.upper - self.lower)


def denormalise_point(x_norm, design_vars: list[DesignVar]):
    x_norm = np.asarray(x_norm, float).ravel()
    assert len(x_norm) == len(design_vars)

    phys = {}
    for xi, dv in zip(x_norm, design_vars):
        phys[dv.name] = dv.denorm(float(xi))
    return phys


def normalise_point(phys_dict: dict, design_vars: list[DesignVar]):
    xs = []
    for dv in design_vars:
        xs.append(dv.norm(float(phys_dict[dv.name])))
    return np.asarray(xs, float)


def df_to_training_data(df: pd.DataFrame, design_vars: list[DesignVar], y_col: str = "failure"):
    if df is None or len(df) == 0:
        raise ValueError("df_to_training_data: dataframe is empty.")

    X_list = []
    for _, row in df.iterrows():
        phys = {dv.name: float(row[dv.name]) for dv in design_vars}
        X_list.append(normalise_point(phys, design_vars))

    X = np.vstack(X_list).astype(float)
    Y = df[y_col].to_numpy(dtype=float).reshape(-1, 1)
    return X, Y


def relative_sequential_difference(a):
    a = np.asarray(a, dtype=float)
    if len(a) < 2:
        return np.array([])

    numerator = np.abs(np.diff(a))
    denom = np.max(a)
    if denom == 0:
        denom = 1.0

    return numerator / denom


def save_model_state(model, kernel_path: Path, prior_mean: float | None, xinterval, design_vars: list[DesignVar]):
    """
    Saves model state needed to resume:
    - kernel name/type
    - variance
    - lengthscale(s)
    - prior mean (constant mean)
    - design var names
    - xinterval bounds
    """
    kern = model.gpc.kern

    kdata = {
        "kernel_name": kern.name,  # e.g. "rbf"
        "input_dim": int(model.f_num),
        "ARD": bool(getattr(kern, "ARD", False)),
        "variance": float(kern.variance.item()),
        "lengthscale": np.asarray(kern.lengthscale).reshape(-1).astype(float).tolist(),
        "prior_mean": None if prior_mean is None else float(prior_mean),
        "design_vars": [dv.name for dv in design_vars],
        "xinterval_lower": np.asarray(xinterval[0], float).reshape(-1).tolist(),
        "xinterval_upper": np.asarray(xinterval[1], float).reshape(-1).tolist(),
    }

    kernel_path = Path(kernel_path)
    kernel_path.parent.mkdir(parents=True, exist_ok=True)
    with open(kernel_path, "w") as f:
        json.dump(kdata, f, indent=2)


def load_model_state(
    kernel_path: Path,
    d: int,
    expected_design_vars: list[str],
    expected_xinterval,
):
    """
    Load state and rebuild an RBF kernel (ARD).

    Returns
    -------
    kernel : GPy.kern.RBF
    saved_prior_mean : float | None
    """
    kernel_path = Path(kernel_path)
    with open(kernel_path, "r") as f:
        kdata = json.load(f)

    saved_vars = kdata.get("design_vars", None)
    if saved_vars is not None and list(saved_vars) != list(expected_design_vars):
        raise ValueError(
            "Design variable order mismatch.\n"
            f"Saved:    {saved_vars}\n"
            f"Expected: {expected_design_vars}"
        )

    xl_saved = np.asarray(kdata.get("xinterval_lower", []), float).reshape(-1)
    xu_saved = np.asarray(kdata.get("xinterval_upper", []), float).reshape(-1)

    xl_exp = np.asarray(expected_xinterval[0], float).reshape(-1)
    xu_exp = np.asarray(expected_xinterval[1], float).reshape(-1)

    if xl_saved.size and (not np.allclose(xl_saved, xl_exp) or not np.allclose(xu_saved, xu_exp)):
        raise ValueError(
            "xinterval mismatch.\n"
            f"Saved:    ({xl_saved}, {xu_saved})\n"
            f"Expected: ({xl_exp}, {xu_exp})"
        )

    kname = str(kdata.get("kernel_name", "")).lower()
    if "rbf" not in kname:
        raise ValueError(
            f"Saved kernel '{kdata.get('kernel_name')}' not supported "
            "(only RBF supported)."
        )

    variance = float(kdata["variance"])
    lengthscale = np.asarray(kdata["lengthscale"], dtype=float).reshape(-1)
    ard = bool(kdata.get("ARD", False))

    if ard:
        if len(lengthscale) != d:
            raise ValueError(
                f"Saved ARD lengthscales have length {len(lengthscale)} but current d={d}."
            )
        kernel = GPy.kern.RBF(input_dim=d, variance=variance, lengthscale=lengthscale, ARD=True)
    else:
        kernel = GPy.kern.RBF(input_dim=d, variance=variance, lengthscale=float(lengthscale[0]), ARD=False)

    saved_prior_mean = kdata.get("prior_mean", None)
    return kernel, saved_prior_mean

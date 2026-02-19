"""
Implementation note / attribution
--------------------------------
This code is based on the code provided in and from:

Zhao, G., Dougherty, E. R., Yoon, B.-J., Alexander, F. J., & Qian, X. (2021).
*Efficient Active Learning for Gaussian Process Classification by Error Reduction*.

Any deviations from the paper are implementation decisions made for this project.
"""

import numpy as np
import torch
from scipy.stats import qmc



def MCSelector(
    func,
    model,
    mc_search_num=1000,
    return_field=False,
    keep_frac: float = 0.01,   
):

    d = model.f_num

    sampler = qmc.Sobol(d=d, scramble=True, seed=None)
    xspace_big = sampler.random(mc_search_num)

    if hasattr(model, "multi_hyper") and model.multi_hyper:
        p1 = np.zeros(pre_mc_num, dtype=float)
        w = 1.0 / len(model.modelset)
        for m in model.modelset:
            p1 += w * m.predict_proba(xspace_big)[:, 1]
    else:
        p1 = model.predict_proba(xspace_big)[:, 1]  

    score = np.abs(p1 - 0.5)
    sort_idx = np.argsort(score)

    keep_k = max(1, int(round(keep_frac * mc_search_num)))
    
    keep_idx = sort_idx[:keep_k]
    xspace = xspace_big[keep_idx] 

    utilitymat = np.zeros(len(xspace), dtype=float) + float("-Inf")

    if hasattr(model, "multi_hyper") and model.multi_hyper:
        for i in range(len(xspace)):
            x = xspace[i:i+1]
            if hasattr(model, "is_real_data") and model.is_real_data:
                if i in model.dataidx:
                    continue
            for m in model.modelset:
                utilitymat[i] += func(x, m)
    else:
        for i in range(len(xspace)):
            x = xspace[i:i+1]
            if hasattr(model, "is_real_data") and model.is_real_data:
                if i in model.dataidx:
                    continue
            utilitymat[i] = func(x, model)

    max_value = np.max(utilitymat)
    max_index = np.random.choice(np.flatnonzero(utilitymat == max_value))
    x_best = xspace[max_index]  # shape (d,)

    if return_field:
        return x_best, float(max_value), xspace, utilitymat

    return x_best, float(max_value)

# def MCSelector(func, model, mc_search_num=1000, return_field=False):
#     """
#     Monte Carlo search over the design space.

#     Parameters
#     ----------
#     func : callable
#         Acquisition function, e.g. U_SMOCU(...).
#         Must accept func(x, model).
#     model : Model
#         Your model wrapper with .XspaceGenerate.
#     mc_search_num : int
#         Number of random candidate points.
#     return_field : bool
#         If True, also return (xspace, utilitymat) for plotting.

#     Returns
#     -------
#     x_best : ndarray
#         Best candidate point (from xspace).
#     max_value : float
#         Acquisition value at x_best.
#     (optionally) xspace : ndarray, shape (mc_search_num, d)
#     (optionally) utilitymat : ndarray, shape (mc_search_num,)
#     """
#     xspace = model.XspaceGenerate(mc_search_num)

#     utilitymat = np.zeros(mc_search_num) + float('-Inf')

#     if hasattr(model, 'multi_hyper') and model.multi_hyper:
#         for i, x in enumerate(xspace):
#             if hasattr(model, 'is_real_data') and model.is_real_data:
#                 if i in model.dataidx:
#                     continue
#             x = xspace[i:i+1]
#             for m in model.modelset:
#                 utilitymat[i] += func(x, m)
#     else:
#         for i, x in enumerate(xspace):
#             if hasattr(model, 'is_real_data') and model.is_real_data:
#                 if i in model.dataidx:
#                     continue
#             x = xspace[i:i+1]  # all inputs should be 2D array
#             utilitymat[i] = func(x, model)

#     # pick the max (break ties randomly)
#     max_value = np.max(utilitymat, axis=None)
#     max_index = np.random.choice(np.flatnonzero(utilitymat == max_value))

#     if hasattr(model, 'is_real_data') and model.is_real_data:
#         model.dataidx = np.append(model.dataidx, max_index)

#     x_best = xspace[max_index]

#     if return_field:
#         return x_best, max_value, xspace, utilitymat

#     return x_best, max_value


def Multi_start_SGD(
    func,
    model,
    mc_search_num: int = 1000,
    learning_rate: float = 1e-3,
    n_starts: int = 5,
    top_frac: float = 0.1,
    n_sgd_steps: int = 200,
    frac_mc_search: float = 0.01,
    return_field: bool = False,
):
    """
    Multi-start SGD for maximising an acquisition function.

    Parameters
    ----------
    func : function
        Acquisition function, e.g. acq = U_SMOCU(...).
        Must accept func(x, model, version='numpy' or 'pytorch').
    model : Model
        The GPC model.
    mc_search_num : int
        Number of random candidates for the initial MC search.
    learning_rate : float
        SGD learning rate.
    n_starts : int
        Number of starting points for SGD.
    top_frac : float
        Fraction of best MC candidates to consider as start pool for the extra starts (besides the MC-best).
    n_sgd_steps : int
        Number of SGD steps per start.
    return_field : bool
        If True, also return (xspace, utilitymat, x_best_mc, acq_best_mc).

    Returns
    -------
    x_best : np.ndarray, shape (d,)
        Best point after multi-start SGD (normalised space).
    acq_best : float
        Acquisition value at x_best (numpy evaluation).
    """
    xl, xu = model.xinterval
    xl = np.asarray(xl, dtype=float).reshape(1, -1)
    xu = np.asarray(xu, dtype=float).reshape(1, -1)

    x_best_mc_2d, acq_best_mc, xspace, utilitymat = MCSelector(
        func, model, mc_search_num=mc_search_num, return_field=True, keep_frac=frac_mc_search,
    )

    xspace = np.asarray(xspace, float)
    utilitymat = np.asarray(utilitymat, float)
    n_cand, d = xspace.shape

    idx_best = int(np.argmax(utilitymat))
    x_best_mc = xspace[idx_best].copy().reshape(-1)  

    n_top = max(1, int(top_frac * n_cand))
    sort_idx = np.argsort(-utilitymat) 
    top_idx = sort_idx[:n_top]

    n_extra = max(0, n_starts - 1)
    if n_extra > 0:
        pool = [i for i in top_idx if i != idx_best]
        if len(pool) == 0:
            extra_indices = []
        else:
            replace = n_extra > len(pool)
            extra_indices = list(np.random.choice(pool, size=n_extra, replace=replace))
        start_indices = [idx_best] + extra_indices
    else:
        start_indices = [idx_best]

    best_x = None
    best_acq = -np.inf

    for idx in start_indices:
        x_start = xspace[idx] 
        x0 = torch.tensor(
            x_start.reshape(1, -1),
            device="cpu",
            requires_grad=True,
            dtype=torch.float64,
        )
        optimizer = torch.optim.SGD([x0], lr=learning_rate)

        xl_t = torch.tensor(xl, device="cpu", dtype=x0.dtype)
        xu_t = torch.tensor(xu, device="cpu", dtype=x0.dtype)

        for _ in range(n_sgd_steps):
            loss = -func(x0, model, version="pytorch")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                x0.clamp_(xl_t, xu_t)

        x_final_2d = x0.detach().cpu().numpy()
        acq_final = float(func(x_final_2d, model, version="numpy"))
        x_final_1d = x_final_2d.reshape(-1)     

        if acq_final > best_acq:
            best_acq = acq_final
            best_x = x_final_1d

    if return_field:
        return best_x, best_acq, x_best_mc, float(acq_best_mc), xspace, utilitymat
    else:
        return best_x, best_acq

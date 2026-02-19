"""
Implementation note / attribution
--------------------------------
This code is based on the code provided in and from:

Zhao, G., Dougherty, E. R., Yoon, B.-J., Alexander, F. J., & Qian, X. (2021).
*Efficient Active Learning for Gaussian Process Classification by Error Reduction*.

Any deviations from the paper are implementation decisions made for this project.

"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import torch
from scipy.stats import qmc


def _predictive_entropy_numpy(model, X):
    """
    Predictive entropy for a batch of points X (numpy).

    Parameters
    ----------
    model : Model
        The GPC model.
    X : ndarray, shape (N, d)
        Normalised input points.

    Returns
    -------
    H : ndarray, shape (N,)
        Entropy values for each row of X.
    """
    X = np.asarray(X, float).reshape(-1, model.f_num)
    py = model.predict_proba(X)
    p1 = py[:, 1]
    eps = 1e-12
    p = np.clip(p1, eps, 1.0 - eps)
    H = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return H



def MCSelector(func, model, mc_search_num = 1000):
    xspace = model.XspaceGenerate(mc_search_num)

    utilitymat = np.zeros(mc_search_num)+float('-Inf')

    if hasattr(model, 'multi_hyper') and model.multi_hyper:
            for i, x in enumerate(xspace):
                if hasattr(model, 'is_real_data') and model.is_real_data:
                    if i in model.dataidx:
                        continue
                x = xspace[i:i+1]
                for m in model.modelset:
                    utilitymat[i]+= func(x, m)
    else:
        for i, x in enumerate(xspace):
            if hasattr(model, 'is_real_data') and model.is_real_data:
                if i in model.dataidx:
                    continue
            x = xspace[i:i+1]# all the inputs should take 2d array 
            # if version == 'pytorch':
            #     x = torch.tensor(x, requires_grad=True)
            utilitymat[i] = func(x, model)
    
    max_value = np.max(utilitymat, axis = None)
    max_index = np.random.choice(np.flatnonzero(utilitymat == max_value))

    if hasattr(model, 'is_real_data') and model.is_real_data:
        model.dataidx = np.append(model.dataidx, max_index)

    # plt.figure()
    # plt.plot(xspace, utilitymat, 'ro')
    # plt.show()
    
    x = xspace[max_index]

    # plt.figure()
    # plt.plot(xspace, utilitymat)
    # plt.show()

    return x, max_value

def RandomSampling(model):
    x = model.XspaceGenerate(1)
    max_value = 0
    return x, max_value

def SGD(func, model, n_candidates=2000, n_steps=200, learning_rate = 0.001):
    #for mm in range(100):
    # random_num = n_candidates
    # random_num = round(0.7*mc_search_num)
    #x11, value11 = MCSelector(func, model, mc_search_num)

    # ------------------------Change ------------------------------
    x1, value1 = MCSelector(func, model, n_candidates)
    # x1, value1, xspace, utilitymat = MCSelector_2(func, model, random_num, return_field=True)
    # ------------------------Change ------------------------------

    
    #x0 = model.XspaceGenerate(1).reshape(-1)
    x0 = torch.tensor(x1, device="cpu", requires_grad= True)
    optimizer = torch.optim.SGD([x0], lr=learning_rate)
    

    # for _ in range(round(0.3*mc_search_num)):
    #     loss = -func(x0, model, version='pytorch')

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     print("loss: {}".format(loss))
    
    # x0 = torch.tensor(x1, requires_grad= True)
    # optimizer = torch.optim.Adam([x0], lr=learning_rate)
    

    # for _ in range(round(0.3*mc_search_num)):
    for _ in range(round(n_steps)):

        loss = -func(x0, model, version='pytorch')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

       # print("loss: {}".format(loss))

    # return x0.detach().numpy(), -loss, x1, value1, xspace, utilitymat
    return x0.detach().numpy(), -loss

    # func2 = lambda x: -1.0*func(x, model)
    # bounds = np.array([model.xinterval[0], model.xinterval[1]])*np.ones((model.f_num, 2))
    # res = minimize(func2, x0, method='TNC', options={'disp':False}, bounds = bounds)
    # xstar = res.x
    # max_value = -res.fun
    # return xstar, max_value
    # max_value = float('-Inf')
    # for mm in range(50):
    #     x0 = model.XspaceGenerate(1).item()
    #     func2 = lambda x: -1.0*func(x, model)
    #     bounds = [(model.xinterval[0], model.xinterval[1])]
    #     res = minimize(func2, x0, method='TNC', 
    #                     options={ 'disp':False}, bounds = bounds)
    #     xstar22 = res.x
    #     max_value22 = -res.fun
    #     print(res)
    #     if max_value22.item() > max_value:
    #         max_value = max_value22.item()
    #         xstar = xstar22



    # # x0 = model.XspaceGenerate(1).item()
    # # func2 = lambda x: -1.0*func(x, model)
    # # bounds = [(-4, 4)]
    # # res = minimize(func2, x0, method='trust-constr', 
    # #                 options={#'xatol':1e-8, 
    # #                 'disp':True}, bounds = bounds)
    # # x = res.x
    # # max_value = -res.fun
    # return xstar, max_value


# def MCSelector_2(func, model, mc_search_num=1000, return_field=False, seed=None):
#     d = model.f_num
#     sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
#     xspace = sampler.random(mc_search_num)  # (mc_search_num, d) in [0,1]

#     utilitymat = np.empty(mc_search_num, dtype=float)
#     for i in range(mc_search_num):
#         x = xspace[i:i+1]              # (1, d)
#         utilitymat[i] = func(x, model) # scalar

#     max_value = float(np.max(utilitymat))
#     max_index = np.random.choice(np.flatnonzero(utilitymat == max_value))
#     x_best = xspace[max_index]

#     if return_field:
#         return x_best, max_value, xspace, utilitymat
#     return x_best, max_value


def MCSelector_3(
    func,
    model,
    mc_search_num=1000,
    return_field=False,
    pre_mc_factor: int = 10,         
    keep_frac: float | None = 0.2,  
    keep_k: int | None = None,      
    score_mode: str = "abs",        
    sobol_seed: int | None = None, 
):

    d = model.f_num

    pre_mc_num = int(pre_mc_factor * mc_search_num)
    sampler = qmc.Sobol(d=d, scramble=True, seed=sobol_seed)
    xspace_big = sampler.random(pre_mc_num)

    if hasattr(model, "multi_hyper") and model.multi_hyper:
        p1 = np.zeros(pre_mc_num, dtype=float)
        w = 1.0 / len(model.modelset)
        for m in model.modelset:
            p1 += w * m.predict_proba(xspace_big)[:, 1]
    else:
        p1 = model.predict_proba(xspace_big)[:, 1]  # shape (pre_mc_num,)

    if score_mode == "abs":
        score = np.abs(p1 - 0.5)
        sort_idx = np.argsort(score)
    elif score_mode == "entropy":
        eps = 1e-12
        p = np.clip(p1, eps, 1.0 - eps)
        score = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
        sort_idx = np.argsort(-score)
    else:
        raise ValueError("score_mode must be 'abs' or 'entropy'.")

    if keep_k is None:
        if keep_frac is None:
            raise ValueError("Provide either keep_frac or keep_k.")
        keep_k = max(1, int(round(keep_frac * pre_mc_num)))
    keep_k = int(min(keep_k, pre_mc_num))

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
    x_best = xspace[max_index]

    if return_field:
        return x_best, float(max_value), xspace, utilitymat

    return x_best, float(max_value)



def MCSelector_2(func, model, mc_search_num=1000, return_field=False):
    """
    Monte Carlo search over the design space.

    Parameters
    ----------
    func : callable
        Acquisition function, e.g. U_SMOCU(...).
        Must accept func(x, model).
    model : Model
        Your model wrapper with .XspaceGenerate.
    mc_search_num : int
        Number of random candidate points.
    return_field : bool
        If True, also return (xspace, utilitymat) for plotting.

    Returns
    -------
    x_best : ndarray, shape (1, d)
        Best candidate point.
    max_value : float
        Acquisition value at x_best.
    (optionally) xspace : ndarray, shape (mc_search_num, d)
    (optionally) utilitymat : ndarray, shape (mc_search_num,)
    """
    # xspace = model.XspaceGenerate(mc_search_num)

    d = model.f_num
    sampler = qmc.Sobol(d=d, scramble=True, seed=None) 
    xspace = sampler.random(mc_search_num)
    
    utilitymat = np.zeros(mc_search_num) + float('-Inf')

    if hasattr(model, 'multi_hyper') and model.multi_hyper:
        for i, x in enumerate(xspace):
            if hasattr(model, 'is_real_data') and model.is_real_data:
                if i in model.dataidx:
                    continue
            x = xspace[i:i+1]
            for m in model.modelset:
                utilitymat[i] += func(x, m)
    else:
        for i, x in enumerate(xspace):
            if hasattr(model, 'is_real_data') and model.is_real_data:
                if i in model.dataidx:
                    continue
            x = xspace[i:i+1]  # all inputs should be 2D array
            utilitymat[i] = func(x, model)

    # pick the max (break ties randomly)
    max_value = np.max(utilitymat, axis=None)
    max_index = np.random.choice(np.flatnonzero(utilitymat == max_value))

    if hasattr(model, 'is_real_data') and model.is_real_data:
        model.dataidx = np.append(model.dataidx, max_index)

    x_best = xspace[max_index]

    if return_field:
        return x_best, max_value, xspace, utilitymat

    return x_best, max_value

def Multi_start_SGD(
    func,
    model,
    mc_search_num: int = 1000,
    learning_rate: float = 1e-3,
    n_starts: int = 5,
    top_frac: float = 0.1,
    n_sgd_steps: int = 200,
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
        Number of starting points for SGD (multi-start).
        NOTE: one of these is always the MC-best candidate.
    top_frac : float
        Fraction of best MC candidates to consider as start pool for the *extra* starts (besides the MC-best).
    n_sgd_steps : int
        Number of SGD steps per start.
    return_field : bool
        If True, also return (xspace, utilitymat, x_best_mc, acq_best_mc).

    Returns
    -------
    x_best : np.ndarray, shape (d,)
        Best point after multi-start SGD (normalised space).
    acq_best : float
        Acquisition value at x_best.
    """
    x_best_mc_2d, acq_best_mc, xspace, utilitymat = MCSelector_3(
        func, model, mc_search_num=mc_search_num, return_field=True, pre_mc_factor=1, keep_frac=0.1
    )

    # x_best_mc_2d, acq_best_mc, xspace, utilitymat = MCSelector_2(
    #     func, model, mc_search_num=mc_search_num, return_field=True
    # )

    xspace = np.asarray(xspace, float)
    utilitymat = np.asarray(utilitymat, float)
    n_cand, d = xspace.shape

    idx_best = int(np.argmax(utilitymat))
    x_best_mc = xspace[idx_best].copy().reshape(-1)  # (d,)

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
            extra_indices = list(
                np.random.choice(pool, size=n_extra, replace=replace)
            )
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
        )
        optimizer = torch.optim.SGD([x0], lr=learning_rate)

        for _ in range(n_sgd_steps):
            loss = -func(x0, model, version="pytorch")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                x0.clamp_(0.0, 1.0)

        x_final_2d = x0.detach().cpu().numpy()        # shape (1, d)
        acq_final = float(func(x_final_2d, model, version="numpy"))
        x_final_1d = x_final_2d.reshape(-1)           # shape (d,)

        if acq_final > best_acq:
            best_acq = acq_final
            best_x = x_final_1d

    if return_field:
        return best_x, best_acq, x_best_mc, float(acq_best_mc), xspace, utilitymat
    else:
        return best_x, best_acq


def hybrid_entropy_smocu_step(
    model,
    acq_smocu,
    sobol_num: int = 5000,
    frac_keep_entropy: float = 0.2,
    n_entropy_steps: int = 100,
    entropy_lr: float = 1e-2,
    p_band: tuple = (0.4, 0.6),
    do_smocu_sgd: bool = True,
    smocu_n_steps: int = 200,
    smocu_lr: float = 1e-3,
    sobol_seed: int | None = None,
):
    """
    One active-learning step that combines:
      1) global Sobol sampling
      2) entropy-based preselection + gradient refinement
      3) probability-band filter
      4) SMOCU-based selection
      5) optional final SMOCU gradient ascent

    Parameters
    ----------
    model : Model
        The GPC model. 
    acq_smocu : function
        Acquisition function for SMOCU, e.g. acq_smocu = U_SMOCU(...).
        Must accept: acq_smocu(x, model, version='numpy' or 'pytorch').
    sobol_num : int
        Number of initial Sobol samples.
    frac_keep_entropy : float
        Fraction of highest-entropy Sobol points to keep for refinement.
    n_entropy_steps : int
        Number of SGD steps per entropy refinement run.
    entropy_lr : float
        Learning rate for entropy-based gradient ascent.
    p_band : tuple
        Probability band (p_low, p_high) for filtering refined points. 
    do_smocu_sgd : bool
        If True, perform final SGD refinement on SMOCU starting from the best band-filtered point. If False, use the band-filter best point directly (no SMOCU gradient step).
    smocu_n_steps : int
        Number of SGD steps for SMOCU refinement.
    smocu_lr : float
        Learning rate for SMOCU SGD.
    sobol_seed : int or None
        Optional Sobol seed. If None, uses randomness for each evaluation. 

    Returns
    -------
    x_star : ndarray, shape (1, d)
        Selected next point in normalised space.
    acq_value : float
        SMOCU acquisition value at x_star.
    """
    d = model.f_num

    # Sobol sampling over the design space
    # -------------------------------------------------------------
    sampler = qmc.Sobol(d=d, scramble=True, seed=sobol_seed)
    X0 = sampler.random(sobol_num)

    # Entropy-based preselection
    # -------------------------------------------------------------
    H0 = _predictive_entropy_numpy(model, X0) 
    idx_sorted = np.argsort(-H0)
    n_keep = max(1, int(frac_keep_entropy * sobol_num))
    idx_keep = idx_sorted[:n_keep]
    X_keep = X0[idx_keep]

    # Gradient ascent on entropy for kept points
    # -------------------------------------------------------------
    X_refined = np.empty_like(X_keep)

    for i, x_start in enumerate(X_keep):
        x0 = torch.tensor(
            x_start.reshape(1, -1), 
            device="cpu",
            dtype=torch.float64,
            requires_grad=True,
        )
        optimizer = torch.optim.SGD([x0], lr=entropy_lr)

        for _ in range(n_entropy_steps):
            py_t = model.predict_proba_torch(x0)  # (1,2)
            p1_t = py_t[:, 1]
            p1_t = torch.clamp(p1_t, 1e-6, 1.0 - 1e-6)
            H_t = -(p1_t * torch.log(p1_t) + (1.0 - p1_t) * torch.log(1.0 - p1_t))
            loss = -H_t.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                x0.clamp_(0.0, 1.0)

        X_refined[i, :] = x0.detach().cpu().numpy().reshape(-1)
 
    # Probability band filter on refined points
    # -------------------------------------------------------------
    py_ref = model.predict_proba(X_refined)
    p1_ref = py_ref[:, 1]
    p_low, p_high = p_band
    mask_band = (p1_ref >= p_low) & (p1_ref <= p_high)

    if np.any(mask_band):
        X_band = X_refined[mask_band]
    else:
        print("[hybrid] Warning: no points in prob band, using max-entropy point.")
        idx_max_H = int(np.argmax(H0[idx_keep]))
        X_band = X_refined[idx_max_H:idx_max_H+1]

    # Evaluate SMOCU on filtered points and choose best
    # -------------------------------------------------------------
    n_band = X_band.shape[0]
    acq_vals = np.empty(n_band, dtype=float)
    for i in range(n_band):
        x_i = X_band[i:i+1] 
        acq_vals[i] = float(acq_smocu(x_i, model, version="numpy"))

    idx_best = int(np.argmax(acq_vals))
    x_best_init = X_band[idx_best:idx_best+1] 

    if not do_smocu_sgd:
        acq_val = float(acq_smocu(x_best_init, model, version="numpy"))
        return x_best_init.reshape(-1), acq_val

    # Optional final gradient ascent on SMOCU
    # -------------------------------------------------------------
    x0 = torch.tensor(
        x_best_init.reshape(1, -1),
        device="cpu",
        dtype=torch.float64,
        requires_grad=True,
    )
    optimizer = torch.optim.SGD([x0], lr=smocu_lr)

    for _ in range(smocu_n_steps):
        loss = -acq_smocu(x0, model, version="pytorch")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x0.clamp_(0.0, 1.0)

    x_final = x0.detach().cpu().numpy().reshape(1, -1)
    acq_val = float(acq_smocu(x_final, model, version="numpy"))

    return x_final.reshape(-1), acq_val
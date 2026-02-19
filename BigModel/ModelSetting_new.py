"""
Implementation note / attribution
--------------------------------
This code is based on the code provided in and from:

Zhao, G., Dougherty, E. R., Yoon, B.-J., Alexander, F. J., & Qian, X. (2021).
*Efficient Active Learning for Gaussian Process Classification by Error Reduction*.

Any deviations from the paper are implementation decisions made for this project.

"""

import math
import numpy as np
import torch
import GPy
from scipy.stats import norm
from scipy import special

A = np.polynomial.hermite.hermgauss(8)


def Xtruncated(xlower, xupper, xspace):
    xspace = np.asarray(xspace, float)
    xlower = np.asarray(xlower, float)
    xupper = np.asarray(xupper, float)
    idx = np.all((xspace >= xlower) & (xspace <= xupper), axis=1)
    return xspace[idx], idx


class Model:
    def __init__(
        self,
        X,
        Y,
        parameters,
        xinterval=None,
        optimize=False,
        kern_variance_fix=False,
        kern_lengthscale_fix=False,
        mean_fix=False,
        vinterval=None,
        linterval=None,
        prior_mean=None,
        px_log=0.0,
    ):
        X = np.asarray(X, float)
        Y = np.asarray(Y, float).reshape(-1, 1)

        self.f_num = X.shape[1]
        self.parameters = parameters
        self.c_num = parameters[1]

        base_kernel = parameters[2]
        kernel = base_kernel.copy()
        lik = parameters[3]

        self.lik = lik
        self.vinterval = vinterval
        self.linterval = linterval
        self.prior_mean = prior_mean
        self.px_log = float(px_log)

        if xinterval is None:
            lower = np.zeros(self.f_num)
            upper = np.ones(self.f_num)
            self.xinterval = (lower, upper)
        else:
            xl, xu = xinterval
            self.xinterval = (np.asarray(xl, float), np.asarray(xu, float))

        if vinterval is not None:
            kernel.variance.constrain_bounded(vinterval[0], vinterval[1])

        if kern_variance_fix:
            kernel.variance.fix()
        else:
            kernel.variance.unfix()

        if linterval is not None:
            kernel.lengthscale.constrain_bounded(linterval[0], linterval[1])

        if kern_lengthscale_fix:
            kernel.lengthscale.fix()
        else:
            kernel.lengthscale.unfix()

        if prior_mean is not None:
            mean = GPy.mappings.Constant(
                input_dim=self.f_num,
                output_dim=1,
                value=prior_mean,
            )
            if mean_fix:
                mean.fix()
            else:
                mean.unfix()
        else:
            mean = None

        m = GPy.core.GP(
            X=X,
            Y=Y,
            kernel=kernel,
            mean_function=mean,
            inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
            likelihood=lik,
        )

        if optimize:
            m.optimize_restarts(
                optimizer="bfgs",
                num_restarts=10,
                max_iters=800,
                verbose=False,
            )
            m = GPy.core.GP(
                X=X,
                Y=Y,
                kernel=kernel,
                mean_function=mean,
                inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                likelihood=lik,
            )

        self.gpc = m
        self.optimize = optimize

    
    def predict_proba(self, x):
        """
        Predict class probabilities using the GPy model.
        
        """
        x = np.asarray(x, float).reshape(-1, self.f_num)
        N = len(x)
        M = N // 1000

        if M <= 1:
            py1 = self.gpc.predict(x)[0]        # shape (N,1)
            py0 = 1.0 - py1
            pymat = np.concatenate((py0, py1), axis=1)
            return pymat
        else:
            pymat = np.zeros((N, 2))
            for m in range(M):
                idx = range(m * 1000, m * 1000 + 1000)
                py1 = self.gpc.predict(x[idx, :])[0]
                pymat[idx, 1:2] = py1
                pymat[idx, 0:1] = 1.0 - py1
            idx = range(m * 1000 + 1000, N)
            py1 = self.gpc.predict(x[idx, :])[0]
            pymat[idx, 1:2] = py1
            pymat[idx, 0:1] = 1.0 - py1
            return pymat

    
    def _noiseless_predict_torch(self, xt):
        woodbury_inv = torch.tensor(self.gpc.posterior.woodbury_inv)
        woodbury_vector = torch.tensor(self.gpc.posterior.woodbury_vector)
        X_ = torch.tensor(self.gpc.X)
        K = self.K

        mu_t = K(xt, X_) @ woodbury_vector
        sigma_tt = K(xt, xt) - K(xt, X_) @ woodbury_inv @ K(X_, xt)

        if self.gpc.mean_function is not None:
            m0 = float(np.asarray(self.gpc.mean_function.C))
            mu_t = mu_t + m0
        
        return mu_t, sigma_tt
        

    def K(self, xt, xs):
        kern = self.gpc.kern
        kname = kern.name.lower()

        var = float(kern.variance.item())
        var = torch.as_tensor(var, dtype=xt.dtype, device=xt.device)

        ell_np = np.asarray(kern.lengthscale).reshape(-1)
        if ell_np.size == 1:
            ell_t = torch.as_tensor(ell_np[0], dtype=xt.dtype, device=xt.device)
            xt_scaled = xt / ell_t
            xs_scaled = xs / ell_t
        else:
            ell_t = torch.as_tensor(ell_np, dtype=xt.dtype, device=xt.device).view(1, -1)
            xt_scaled = xt / ell_t
            xs_scaled = xs / ell_t

        r = torch.cdist(xt_scaled, xs_scaled)

        if "rbf" in kname:
            Kts = var * torch.exp(-0.5 * r**2)
        elif "mat32" in kname:
            sqrt3_r = math.sqrt(3.0) * r
            Kts = var * (1.0 + sqrt3_r) * torch.exp(-sqrt3_r)
        else:
            raise NotImplementedError(
                f"Kernel '{kern.name}' not supported in Model.K; "
                "supported: RBF, Matern32."
            )

        return Kts

    def predict_proba_torch(self, xt):
        assert isinstance(xt, torch.Tensor)
        xt = xt.reshape(-1, self.f_num)

        Phi = lambda x: 0.5 * (torch.erf(x / math.sqrt(2.0)) + 1.0)
        mu_t, sigma_tt = self._noiseless_predict_torch(xt)
        ft_hat = mu_t / torch.sqrt(sigma_tt + 1.0)

        py1 = Phi(ft_hat)
        py0 = 1.0 - py1
        pymat = torch.cat((py0, py1), dim=1)
        return pymat

    def _calculate_mean_and_variance(self, xt, xs):
        xt = np.asarray(xt, float).reshape(-1, self.f_num)
        xs = np.asarray(xs, float).reshape(-1, self.f_num)

        muvar = self.gpc.predict_noiseless(np.concatenate((xs, xt)), full_cov=False)
        mu = muvar[0].reshape(-1)
        var = muvar[1].reshape(-1)

        mu_s = mu[:-1]
        mu_t = mu[-1]
        sigma_ss = var[:-1]

        X_ = self.gpc.X
        sigma_st = (
            self.gpc.kern.K(xs, xt)
            - self.gpc.kern.K(xs, X_) @ self.gpc.posterior.woodbury_inv @ self.gpc.kern.K(X_, xt)
        ).reshape(-1)
        sigma_tt = var[-1]
        sigma_tt_hat = sigma_tt - sigma_st**2 / sigma_ss
        return mu_s, mu_t, sigma_ss, sigma_st, sigma_tt_hat

    def _calculate_mean_and_variance_torch(self, x1, x2):
        xt = x1.reshape(-1, self.f_num)
        xs = torch.tensor(x2.reshape(-1, self.f_num))

        X_ = torch.tensor(self.gpc.X)
        woodbury_inv = torch.tensor(self.gpc.posterior.woodbury_inv)
        woodbury_vector = torch.tensor(self.gpc.posterior.woodbury_vector)

        muvar = self.gpc.predict_noiseless(x2, full_cov=False)
        mu_s = torch.tensor(muvar[0])
        sigma_ss = torch.tensor(muvar[1])

        K = self.K
        mu_t, sigma_tt = self._noiseless_predict_torch(xt)
        sigma_st = K(xs, xt) - K(xs, X_) @ woodbury_inv @ K(X_, xt)
        sigma_tt_hat = sigma_tt - sigma_st**2 / sigma_ss

        mu_s = mu_s.reshape(-1)
        mu_t = mu_t.reshape(-1)
        sigma_ss = sigma_ss.reshape(-1)
        sigma_st = sigma_st.reshape(-1)
        sigma_tt_hat = sigma_tt_hat.reshape(-1)
        return mu_s, mu_t, sigma_ss, sigma_st, sigma_tt_hat

    def _calculate_posterior_predictive_from_joint_distribution(self, xt, xs, pt1s1, version="numpy"):
        if version == "pytorch":
            pt = self.predict_proba_torch(xt)
        else:
            pt = self.predict_proba(xt)

        assert pt.shape == (1, 2)
        pt0, pt1 = pt[0, 0], pt[0, 1]

        ps = self.predict_proba(xs)
        ps0, ps1 = ps[:, 0], ps[:, 1]

        if version == "pytorch":
            ps1 = torch.tensor(ps1)

        pt0s1 = ps1 - pt1s1

        ps1_t1 = pt1s1 / pt1
        ps0_t1 = 1.0 - ps1_t1
        ps1_t0 = pt0s1 / pt0
        ps0_t0 = 1.0 - ps1_t0

        if version == "pytorch":
            column_stack = torch.column_stack
        else:
            column_stack = np.column_stack

        ps_t0 = column_stack((ps0_t0, ps1_t0))
        ps_t1 = column_stack((ps0_t1, ps1_t1))
        return ps_t0, ps_t1

    def OneStepPredict(self, xt, xs, version="numpy"):
        if version == "pytorch":
            calculate_mean_variance = self._calculate_mean_and_variance_torch
            erf = torch.erf
            sqrt = torch.sqrt
            zeros = torch.zeros
        else:
            calculate_mean_variance = self._calculate_mean_and_variance
            erf = special.erf
            sqrt = np.sqrt
            zeros = np.zeros

        mu_s, mu_t, sigma_ss, sigma_st, sigma_tt_hat = calculate_mean_variance(xt, xs)
        sigma_s = sqrt(sigma_ss)
        Phi = lambda x: 0.5 * (erf(x / math.sqrt(2.0)) + 1.0)

        def func4(f0):
            f0 = float(f0)
            fs = f0 * math.sqrt(2.0) * sigma_s + mu_s
            mu_t_hat = mu_t + sigma_st / sigma_ss * (fs - mu_s)
            ft_hat = mu_t_hat / sqrt(sigma_tt_hat + 1.0)
            term1 = Phi(ft_hat)
            term2 = Phi(fs)
            return term1 * term2 / math.sqrt(math.pi)

        pt1s1 = zeros(len(xs))
        for i, f0 in enumerate(A[0]):
            pt1s1 += func4(f0) * A[1][i]

        ps_t0, ps_t1 = self._calculate_posterior_predictive_from_joint_distribution(
            xt, xs, pt1s1, version=version
        )
        return ps_t0, ps_t1

    def DataApprox(self, x):
        anum = 3
        X = self.gpc.X
        Y = self.gpc.Y
        kernel = self.parameters[2]

        d_ls = np.asarray(kernel.lengthscale).reshape(-1)
        if d_ls.size == 1:
            d_vec = np.full(self.f_num, float(d_ls[0]))
        else:
            d_vec = d_ls

        l = anum * d_vec
        xl, xu = self.xinterval
        x = np.asarray(x, float).reshape(-1, self.f_num)

        xlower = np.maximum(xl, x - l)
        xupper = np.minimum(xu, x + l)

        Xtr, idxarray = Xtruncated(xlower, xupper, X)
        Ytr = Y[idxarray].reshape(-1, 1)
        return Xtr, Ytr

    def UpdateNew(self, x, y):
        x = np.asarray(x, float).reshape(-1, self.f_num)
        X = np.concatenate((self.gpc.X, x), axis=0)
        Y = np.concatenate((self.gpc.Y, [[y]]), axis=0)
        model2 = self.ModelTrain(X, Y)
        return model2

    def ModelTrain(
        self,
        X,
        Y,
        optimize=False,
        kern_variance_fix=False,
        kern_lengthscale_fix=False,
        mean_fix=False,
    ):
        parameters = self.parameters.copy()
        parameters[2] = self.gpc.kern
        model2 = Model(
            X,
            Y,
            parameters=parameters,
            xinterval=self.xinterval,
            optimize=optimize,
            kern_variance_fix=kern_variance_fix,
            kern_lengthscale_fix=kern_lengthscale_fix,
            mean_fix=mean_fix,
            vinterval=self.vinterval,
            linterval=self.linterval,
            prior_mean=self.prior_mean,
            px_log=self.px_log,
        )
        return model2

    def training_misclassification(self, X=None, Y=None, gp=None):
        if gp is None:
            gp = self.gpc
        if X is None:
            X = gp.X
        if Y is None:
            Y = gp.Y

        X = np.asarray(X, float)
        y_true = np.asarray(Y, float).ravel().astype(int)

        p1 = gp.predict(X)[0].ravel()
        y_pred = (p1 >= 0.5).astype(int)
        return float(np.mean(y_pred != y_true))

    def Update(
        self,
        x,
        y,
        optimize=False,
        kern_variance_fix=False,
        kern_lengthscale_fix=False,
        mean_fix=False,
        ll_tol=0.0,
    ):
        x = np.asarray(x, float).reshape(-1, self.f_num)
        X = np.concatenate((self.gpc.X, x))
        Y = np.concatenate((self.gpc.Y, [[y]]), axis=0)

        kernel = self.gpc.kern

        if self.vinterval is not None:
            kernel.variance.constrain_bounded(self.vinterval[0], self.vinterval[1])
        if kern_variance_fix:
            kernel.variance.fix()
        else:
            kernel.variance.unfix()

        if self.linterval is not None:
            kernel.lengthscale.constrain_bounded(self.linterval[0], self.linterval[1])
        if kern_lengthscale_fix:
            kernel.lengthscale.fix()
        else:
            kernel.lengthscale.unfix()

        if self.prior_mean is not None:
            mean = GPy.mappings.Constant(
                input_dim=self.f_num,
                output_dim=1,
                value=self.prior_mean,
            )
            if mean_fix:
                mean.fix()
            else:
                mean.unfix()
        else:
            mean = None

        m = GPy.core.GP(
            X=X,
            Y=Y,
            kernel=kernel,
            mean_function=mean,
            inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
            likelihood=self.lik,
        )

        ll_base = float(m.log_likelihood())
        print("Log_likelihood before optimisation:", ll_base)

        best_model = m
        best_ll = ll_base

        if optimize:
            kernel_opt = kernel.copy()
            print(kernel_opt)

            m_opt = GPy.core.GP(
                X=X,
                Y=Y,
                kernel=kernel_opt,
                mean_function=mean,
                inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                likelihood=self.lik,
            )

            m_opt.optimize_restarts(
                optimizer="bfgs",
                num_restarts=6,
                max_iters=500,
                verbose=False,
            )

            print("Optimised variance:", m_opt.kern.variance)
            print("Optimised lengthscale:", m_opt.kern.lengthscale)

            m_opt = GPy.core.GP(
                X=X,
                Y=Y,
                kernel=kernel_opt,
                mean_function=mean,
                inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                likelihood=self.lik,
            )

            ll_opt = float(m_opt.log_likelihood())
            print("Log_likelihood after optimisation:", ll_opt)

            ll_frac = ((ll_opt - ll_base) / max(abs(ll_base), 1e-12))
            print("Fractional improvement of lml:", ll_frac)

            if ll_frac > ll_tol:
                best_model = m_opt
                best_ll = ll_opt
                print("Kernel hyperparameters accepted (optimised).")

        self.gpc = best_model

    def XspaceGenerate(self, x_num):
        xl, xu = self.xinterval
        xspace = np.random.uniform(xl, xu, size=(x_num, self.f_num))
        return xspace

    def XspaceGenerateApprox(self, x_num, x):
        ls = np.asarray(self.gpc.kern.lengthscale)

        if ls.size == 1:
            d_vec = np.full(self.f_num, float(ls))
        else:
            d_vec = ls.reshape(-1)

        x = np.asarray(x, float).reshape(1, self.f_num)
        xl, xu = self.xinterval

        xspace = norm.rvs(size=(x_num, self.f_num), loc=x, scale=d_vec)
        xspace, _ = Xtruncated(xl, xu, xspace)

        if len(xspace) == 0:
            # fallback: uniform sampling
            xspace = np.random.uniform(xl, xu, (x_num, self.f_num))
            wspace_log = np.zeros(x_num) + self.px_log
            return xspace, wspace_log, self.px_log

        wspace_log_array = norm.logpdf(xspace, loc=x, scale=d_vec)
        wspace_log = np.sum(wspace_log_array, axis=1) + np.log(x_num / len(xspace))

        return xspace, wspace_log, self.px_log

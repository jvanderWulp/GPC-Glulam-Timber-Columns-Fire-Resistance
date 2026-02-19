"""
Implementation note / attribution
--------------------------------
This code is based on the code provided in and from:

Zhao, G., Dougherty, E. R., Yoon, B.-J., Alexander, F. J., & Qian, X. (2021).
*Efficient Active Learning for Gaussian Process Classification by Error Reduction*.

Any deviations from the paper are implementation decisions made for this project.
"""

import torch


import numpy as np
from numpy.random import choice
from matplotlib import pyplot as plt
import GPy
from scipy.stats import norm, multivariate_normal
from scipy.stats import truncnorm
from InitialSetting import *
from scipy import special
from scipy import integrate
import math
import sys, os
# sys.path.append(os.path.dirname(sys.path[0]))
from svgd import *

import copy 

A = np.polynomial.hermite.hermgauss(8)


def GroundTruthFunction(x):
    # x is a single point f_num
    py = GroundTruthProbability(x)
    py = py.reshape(-1)
    y = choice(range(c_num), p = py)
    return y

def XspaceGenerate_(x_num):
    # xspace = np.linspace(xinterval[0], xinterval[1], x_num)
    # xspace = xspace.reshape(-1, f_num) 
    # xspace = np.random.uniform(xinterval[0], xinterval[1], (x_num, f_num))
    if discrete_label:
        if x_num == len(xspace):
            return xspace
        sampleidx = choice(range(xspace.shape[0]), x_num, replace = False)#############################
        return xspace[sampleidx]
    xspace = np.random.uniform(xinterval[0], xinterval[1], (x_num, f_num))
    return xspace



def XspaceGenerateApprox_(x_num, x):###############How about high dimension
    
    d = kernel.lengthscale
    #xspace = np.random.multivariate_normal(mean = x, scale = d, x_num)
    if f_num > 1:
        cov = np.eye(f_num)*d
        mean = x.reshape(-1)
        xspace = np.random.multivariate_normal(mean = mean, cov = cov, size = x_num)
    else:
        xspace = np.random.normal(x, d, (x_num, 1))##this is only for single dimension
    # idxarray = np.all(xspace >= xinterval[0], axis=1) & np.all(xspace <= xinterval[1], axis = 1)
    # xspace = xspace[idxarray].reshape(-1, f_num)
    xspace = norm.rvs(size = (x_num, f_num), loc = x, scale = d)
    xspace, _ = Xtruncated(xinterval[0], xinterval[1], xspace)
    wspace_log_array = norm.logpdf(xspace, loc=x, scale=d)
    wspace_log = np.sum(wspace_log_array, axis=1)+np.log(x_num/len(xspace))
    
    
    return xspace, wspace_log, px_log

def InitialDataGenerator(f, initial_num = 10): 
    X_ = XspaceGenerate_(initial_num)
    #X_ = np.random.uniform(xinterval[0], xinterval[1], (initial_num, f_num))
    Y_ = np.zeros((initial_num, 1))
    for i in range(initial_num):
        Y_[i] = f(X_[i:i+1])
    
    Xindex = None
    return X_, Y_, Xindex

try:
    BayesianError
except:
    def BayesianError(x_num):
        
        xspace = XspaceGenerate_(x_num)

        pymat = GroundTruthProbability(xspace)
        bayesian_error = np.amin(1-pymat, axis =1)
        return bayesian_error.mean()


def Xtruncated(xlower, xupper, xspace):
    idxarray = np.all(xspace >= xlower, axis=1) & np.all(xspace <= xupper, axis = 1)
    return xspace[idxarray].reshape(-1, f_num), idxarray



class ModelSet():
    def __init__(self,  X, Y, parameters = parameters_, hypernum = 10):
        #__slots__ = ['hypernum', 'f_num', 'X', 'Y', 'multi_hyper', 'hyperset']
        #self.parameterset = parameter# random generate parameterset
        
        self.f_num = f_num
        self.hypernum = hypernum
        self.X = X
        self.Y = Y
        self.multi_hyper = True
        #self.is_real_data = False

        #generate prior set
        varianceset = np.random.uniform(vinterval[0], vinterval[1], [hypernum, 1])####################
        lengthset = np.random.uniform(linterval[0], linterval[1], [hypernum, 1])################
        hyperset0 = np.concatenate((varianceset, lengthset), axis = 1)

        #posterior set
        self.hyperset = self.HyperParticle(hyperset0, n_iter = 1000)

        self.ModelSetGen(self.hyperset)


    def ModelSetGen(self, hyperset):
        self.modelset = []
        for m in range(self.hypernum):
            variance = hyperset[m, 0]
            lengthscale = hyperset[m, 1]
            parameters = parameters_
            parameters[2]=GPy.kern.RBF(f_num, variance=variance, lengthscale=lengthscale)
            self.modelset.append(Model(self.X, self.Y, parameters=parameters, optimize=False))

    def Update(self, x, y):
        x = x.reshape(-1, self.f_num)
        self.X = np.concatenate((self.X, x))
        self.Y = np.concatenate((self.Y, [[y]]), axis = 0)
        self.hyperset = self.HyperParticle(self.hyperset, n_iter = 100)
        self.ModelSetGen(self.hyperset)

    def dloglikelihood(self, theta_array):
        grad_array = np.zeros(theta_array.shape)
        for i, theta in enumerate(theta_array):
            var = theta[0]
            length = theta[1]
            if var < vinterval[0] or var > vinterval[1] or length < linterval[0] or length > linterval[1]:
                if var < vinterval[0]:
                    grad_array[i, 0] = 10
                if var > vinterval[1]:
                    grad_array[i, 0] = -10
                if length < linterval[0]:
                    grad_array[i, 1] = 10
                if length > linterval[1]:
                    grad_array[i, 1] = -10
            else:
                kernel = GPy.kern.RBF(f_num, variance = var, lengthscale = length)
                m = GPy.core.GP(X=self.X,
                                Y=self.Y,
                                kernel=kernel,
                                inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                                likelihood=lik)
                grad_array[i]=-m.objective_function_gradients()[1]

        return grad_array

    def HyperParticle(self, initial_array, n_iter = 1000):
        #array size: particle_num * parameter_num
        updated_array = SVGD().update(initial_array, self.dloglikelihood, n_iter=n_iter, stepsize=0.05)
        return updated_array

    def ObcClassifierError(self, x_num):
        xspace = self.XspaceGenerate(x_num)
        p = 1.0/self.hypernum
        pyTheta = np.zeros((x_num, c_num))
        for model in self.modelset:
            pyTheta += model.predict_proba(xspace)*p
        yhat = np.argmax(pyTheta, axis = 1)
        py = GroundTruthProbability(xspace)
        classifier_error = 1-py[np.arange(x_num), yhat]
        return classifier_error.mean()

    def XspaceGenerate(self, x_num):
        xspace = XspaceGenerate_(x_num)
        return xspace


        


class Model():

    def __init__(self,  X, Y, parameters = parameters_, optimize = False, kern_variance_fix=False, kern_lengthscale_fix=False, mean_fix=False):#######################
        #__slots__ = ['parameters', 'f_num', 'gpc', 'xinterval', 'optimize', 'lik', 'c_num']
        # X size is x_num*fnum
        # Y size is x_num
        self.parameters = parameters
        self.f_num = f_num
        self.c_num = parameters[1]
        kernel = parameters[2].copy()      # ---------------- Change: .copy() ------------------------------
        #lik = parameters[3]
        #self.gpc = GaussianProcessClassifier(kernel = kernel, optimizer=None).fit(X, Y)

        # --------------------------------- Change ----------------------------------------------
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
            mean = GPy.mappings.Constant(input_dim=f_num, output_dim=1, value=prior_mean)
            if mean_fix:
                mean.fix()   # keep the prior mean fixed during optimisation
            else:
                mean.unfix()
        else:
            mean = None
        # ------------------------------------------------------------------------------------
        
        m = GPy.core.GP(X=X,
                        Y=Y, 
                        kernel=kernel,
                        mean_function=mean,            # ----------------- Change ------------------- #
                        inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                        likelihood=lik)
        if optimize:
            m.optimize_restarts(optimizer='bfgs', num_restarts = 10, max_iters=800, verbose=False)

            m = GPy.core.GP(X=X,
                            Y=Y,
                            kernel=kernel,
                            mean_function=mean,            # ----------------- Change ------------------- #
                            inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                            likelihood=lik)
        
            
        # if m.kern.lengthscale.item()>4:
        #     kernel = GPy.kern.RBF(f_num, variance = m.kern.variance.item(), lengthscale = 4)

        #     # ----------------- Change ------------------- #
        #     mean = GPy.mappings.Constant(input_dim=f_num, output_dim=1, value=prior_mean)
        #     mean.fix()
        #     # -------------------------------------------- #
            
        #     m = GPy.core.GP(X = X,
        #                 Y = Y,
        #                 kernel = kernel,
        #                 mean_function=mean,         # ----------------- Change ------------------- #
        #                 inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
        #                 likelihood = lik)
        self.gpc = m
        #self.c_num = c_num
        self.xinterval = xinterval
        self.optimize = optimize
        self.lik = lik
    
    def predict_proba(self, x):
        x = x.reshape(-1, self.f_num)
        M = len(x)//1000
        if M <= 1:
            py_1 = self.gpc.predict(x)[0]
            py_0 = 1 - py_1
            pymat = np.concatenate((py_0, py_1), axis = 1) #pymat size x_num*cnum
            return pymat
        else:
            pymat1 = np.zeros((len(x), 2))
            for m in range(M):
                idx = range(m*1000, m*1000+1000)
                pymat1[idx, 1:2] = self.gpc.predict(x[idx, :])[0]
                pymat1[idx, 0] = 1-pymat1[idx, 1]
            idx = range(m*1000+1000, len(x))
            pymat1[idx, 1:2] = self.gpc.predict(x[idx, :])[0]
            pymat1[idx, 0] = 1-pymat1[idx, 1]

            return pymat1

    # def _noiseless_predict_torch(self, xt):
    #     woodbury_inv = torch.tensor(self.gpc.posterior.woodbury_inv)
    #     woodbury_vector = torch.tensor(self.gpc.posterior.woodbury_vector)
    #     X_ = torch.tensor(self.gpc.X)
    #     K = self.K
        
    #     mu_t = K(xt, X_)@woodbury_vector
    #     sigma_tt = K(xt, xt) - K(xt, X_)@woodbury_inv@K(X_, xt)

    #     return mu_t, sigma_tt

    def _noiseless_predict_torch(self, xt):
        xt = xt.reshape(-1, self.f_num)
        assert isinstance(xt, torch.Tensor)
    
        woodbury_inv = torch.tensor(self.gpc.posterior.woodbury_inv, dtype=xt.dtype, device=xt.device)
        woodbury_vector = torch.tensor(self.gpc.posterior.woodbury_vector, dtype=xt.dtype, device=xt.device)
        X_ = torch.tensor(self.gpc.X, dtype=xt.dtype, device=xt.device)
    
        K = self.K
    
        # Latent GP part
        mu_t = K(xt, X_) @ woodbury_vector
        sigma_tt = K(xt, xt) - K(xt, X_) @ woodbury_inv @ K(X_, xt)
    
        # ---- ADD THE MEAN FUNCTION m(x) ----
        if self.gpc.mean_function is not None:
            # For GPy.mappings.Constant, the constant is stored in .C
            m0 = float(np.asarray(self.gpc.mean_function.C))
            mu_t = mu_t + m0
    
        return mu_t, sigma_tt


    # ----------------------------------------------------------------------- Change ------------------------------------------------------------------------------------#    
    # def K(self, xt, xs):

    #     kern = self.gpc.kern
    #     assert (kern.name == 'rbf') #the function is only coded for rbf kernel
            
    #     l1 = kern.lengthscale.item()
    #     l2 = kern.variance.item()
    #     Kts = l2*torch.exp(-torch.cdist(xt, xs)**2/2/l1**2)
    #     return Kts

    def K(self, xt, xs):
        """
        Torch implementation of the covariance matrix K(xt, xs).
    
        Supports at least:
          - RBF kernel (squared–exponential), with or without ARD
          - Matern 3/2, with or without ARD
    
        Parameters
        ----------
        xt : torch.Tensor, shape (N, d)
        xs : torch.Tensor, shape (M, d)
    
        Returns
        -------
        Kts : torch.Tensor, shape (N, M)
        """
        kern = self.gpc.kern
        kname = kern.name.lower()
    
        # ----- variance as torch scalar (constant w.r.t. xt) -----
        var = float(kern.variance.item())
        var = torch.as_tensor(var, dtype=xt.dtype, device=xt.device)
    
        # ----- lengthscales: handle isotropic and ARD -----
        # GPy stores lengthscale as an array even for isotropic kernels
        ell_np = np.asarray(kern.lengthscale).reshape(-1)  # shape (L,)
        if ell_np.size == 1:
            # isotropic: one lengthscale for all dims
            ell_t = torch.as_tensor(ell_np[0], dtype=xt.dtype, device=xt.device)
            xt_scaled = xt / ell_t
            xs_scaled = xs / ell_t
        else:
            # ARD: one lengthscale per input dimension
            # ell_t: shape (1, d) to broadcast across rows
            ell_t = torch.as_tensor(ell_np, dtype=xt.dtype, device=xt.device).view(1, -1)
            xt_scaled = xt / ell_t          # (N, d)
            xs_scaled = xs / ell_t          # (M, d)
    
        # pairwise distance in *scaled* space
        r = torch.cdist(xt_scaled, xs_scaled)   # shape (N, M)
    
        if "rbf" in kname:
            # Squared–exponential: k = var * exp(-0.5 r^2)
            Kts = var * torch.exp(-0.5 * r**2)
    
        elif "mat32" in kname:
            # Matérn 3/2: k = var * (1 + sqrt(3) r) * exp(-sqrt(3) r)
            sqrt3_r = math.sqrt(3.0) * r
            Kts = var * (1.0 + sqrt3_r) * torch.exp(-sqrt3_r)
    
        else:
            raise NotImplementedError(
                f"Kernel '{kern.name}' not supported in Model.K. "
                "Currently supports RBF (squared–exp) and Matern32."
            )
    
        return Kts

        
        # """
        # Torch implementation of the covariance matrix K(xt, xs).

        # Supports at least:
        # - RBF kernel
        # - Matern32 kernel

        # xt, xs: torch.Tensor with shape (N, d) and (M, d).
        # """
        # kern = self.gpc.kern
        # # GPy uses a .name attribute like 'rbf', 'Matern32', etc.
        # kname = kern.name.lower()

        # # lengthscale and variance are 1D parameters in these kernels
        # ell = float(kern.lengthscale.item())
        # var = float(kern.variance.item())

        # # pairwise distances r = ||xt - xs||
        # r = torch.cdist(xt, xs)

        # if "rbf" in kname:
        #     # RBF: k(r) = var * exp( - r^2 / (2 ell^2) )
        #     Kts = var * torch.exp(-(r ** 2) / (2.0 * ell ** 2))

        # elif "mat32" in kname:
        #     # Matern 3/2: k(r) = var * (1 + sqrt(3) r / ell) * exp(-sqrt(3) r / ell)
        #     sqrt3_r_over_l = math.sqrt(3.0) * r / ell
        #     Kts = var * (1.0 + sqrt3_r_over_l) * torch.exp(-sqrt3_r_over_l)

        # else:
        #     raise NotImplementedError(
        #         f"Kernel '{kern.name}' not supported in Model.K. "
        #         "Currently supports RBF and Matern32."
        #     )

        # return Kts
        # ----------------------------------------------------------------------- Change ------------------------------------------------------------------------------------#    

    
    def predict_proba_torch(self, xt):
        xt = xt.reshape(-1, self.f_num)
        assert(type(xt)==torch.Tensor)
        
        Phi = lambda x : 0.5*(torch.erf(x/math.sqrt(2))+1) 
        mu_t, sigma_tt = self._noiseless_predict_torch(xt)

        ft_hat = mu_t/torch.sqrt(sigma_tt+1)
        py_1 = Phi(ft_hat)
        py_0 = 1 - py_1
        pymat = torch.cat((py_0, py_1), axis = 1)
        return pymat

    def _calculate_mean_and_variance(self, xt, xs):
        xt = xt.reshape(-1, self.f_num)
        xs = xs.reshape(-1, self.f_num)
        muvar = self.gpc.predict_noiseless(np.concatenate((xs, xt)), full_cov=False) 
        mu = muvar[0].reshape(-1)
        mu_s = mu[0:-1]
        mu_t = mu[-1]

        var = muvar[1].reshape(-1)
        sigma_ss = var[0:-1]
        X_ = self.gpc.X
        sigma_st =  self.gpc.kern.K(xs, xt) - self.gpc.kern.K(xs,X_)@self.gpc.posterior.woodbury_inv@self.gpc.kern.K(X_, xt)
        sigma_st = sigma_st.reshape(-1)
        sigma_tt = var[-1]
        sigma_tt_hat = sigma_tt - sigma_st**2/sigma_ss
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
        sigma_st = K(xs, xt) - K(xs, X_)@woodbury_inv@K(X_, xt)
        sigma_tt_hat = sigma_tt - sigma_st**2/sigma_ss

        # muvar = self.gpc.predict_noiseless(np.concatenate((xs, xt)), full_cov=False) 
        # mu = muvar[0].reshape(-1)
        # mu_s = mu[0:-1]
        # mu_t = mu[-1]

        # var = muvar[1].reshape(-1)
        # sigma_ss = var[0:-1]
        # X_ = self.gpc.X
        # sigma_st =  self.gpc.kern.K(xs, xt) - self.gpc.kern.K(xs,X_)@self.gpc.posterior.woodbury_inv@self.gpc.kern.K(X_, xt)
        # sigma_st = sigma_st.reshape(-1)
        # sigma_tt = var[-1]
        # sigma_tt_hat = sigma_tt - sigma_st**2/sigma_ss
        mu_s = mu_s.reshape(-1)
        mu_t = mu_t.reshape(-1)
        sigma_ss = sigma_ss.reshape(-1)
        sigma_st = sigma_st.reshape(-1)
        sigma_tt_hat = sigma_tt_hat.reshape(-1)
        return mu_s, mu_t, sigma_ss, sigma_st, sigma_tt_hat
    
    def _calculate_posterior_predictive_from_joint_distribution(self, xt, xs, pt1s1, version = 'numpy'):
        if version == 'pytorch':
            pt = self.predict_proba_torch(xt)
        else:
            pt = self.predict_proba(xt)
        
        assert(pt.shape == (1, 2))
        pt0, pt1 = pt[0, 0], pt[0, 1]
        ps = self.predict_proba(xs)
        ps0, ps1 = ps[:, 0], ps[:, 1]
        if version == 'pytorch':
            ps1 = torch.tensor(ps1)
        pt0s1 = ps1 - pt1s1
        ps1_t1 = pt1s1/pt1
        ps0_t1 = 1-ps1_t1
        ps1_t0 = pt0s1/pt0
        ps0_t0 = 1-ps1_t0

        if version == 'pytorch':
            column_stack = torch.column_stack
        else:
            column_stack = np.column_stack
        ps_t0 = column_stack((ps0_t0, ps1_t0))
        ps_t1 = column_stack((ps0_t1, ps1_t1))

        return ps_t0, ps_t1

    
    def OneStepPredict(self, xt, xs, version = 'numpy'):
        #xt is xstar, xt is tensor for version = 'pytorch'
        #xs is an array of size x_num*f_num

        if version == 'pytorch':
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
        Phi = lambda x : 0.5*(erf(x/math.sqrt(2))+1) 
        
        def func4(f0):
            #This function use hermite Gaussian quadrature, 
            # return: a x_num array of value with index corresponding to xs. 
            # term3 = 1/(sigma1*math.sqrt(2*math.pi))*math.exp(-0.5*(x3)**2) is normalized as Gaussian function
            f0 = float(f0)
            fs = f0*math.sqrt(2)*sigma_s+mu_s
            mu_t_hat = mu_t + sigma_st/sigma_ss*(fs-mu_s)
            ft_hat = mu_t_hat/sqrt(sigma_tt_hat+1)
            term1 = Phi(ft_hat)
            term2 = Phi(fs)
            return term1*term2/math.sqrt(math.pi)#math.sqrt(math.pi) is the constant for normalized Gaussian

        # joint distribution
        pt1s1 = zeros(len(xs))
        for i, f0 in enumerate(A[0]):
            pt1s1 += func4(f0)*A[1][i]

        ps_t0, ps_t1 = self._calculate_posterior_predictive_from_joint_distribution(xt, xs, pt1s1, version = version)
        return ps_t0, ps_t1


    def DataApprox(self, x):
        anum = 3
        X = self.gpc.X
        Y = self.gpc.Y
        kernel = self.parameters[2]
        d = kernel.lengthscale
        l = anum*d
        xlower = np.maximum(self.xinterval[0], x-l)
        xupper = np.minimum(self.xinterval[1], x+l)
        # idxarray = np.all(X >= xlower, axis=1) & np.all(X <= xupper, axis=1)
        
        X, idxarray = Xtruncated(xlower, xupper, X)

        #X = X[idxarray].reshape(-1, self.f_num)
        Y = Y[idxarray].reshape(-1, 1)

        return X, Y


    def UpdateNew(self, x, y):##############################################################
        x = x.reshape(-1, self.f_num)
        X = np.concatenate((self.gpc.X, x), axis = 0)
        Y = np.concatenate((self.gpc.Y, [[y]]), axis = 0)
        # parameters = self.parameters
        # parameters[2] = self.gpc.kern
        # model2 = Model(X, Y, parameters, optimize = False)
        model2 = self.ModelTrain(X, Y)
        return model2

    def ModelTrain(self, X, Y, optimize=False, kern_variance_fix=False, kern_lengthscale_fix=False, mean_fix=False):
        parameters = self.parameters
        parameters[2] = self.gpc.kern
        model2 = Model(X, Y, parameters, optimize=optimize, kern_variance_fix=kern_variance_fix, kern_lengthscale_fix=kern_lengthscale_fix, mean_fix=mean_fix)
        return model2
    

    def training_misclassification(self, X=None, Y=None, gp=None):
        """
        Misclassification rate on (X, Y).

        - If gp is None -> use this Model's GP (self.gpc).
        - If X, Y are None -> use the GP's own training data (gp.X, gp.Y).
        """
        if gp is None:
            gp = self.gpc
        if X is None:
            X = gp.X
        if Y is None:
            Y = gp.Y

        X = np.asarray(X, float)
        y_true = np.asarray(Y).ravel().astype(int)

        # For Bernoulli GPy model: predict(X)[0] = p(y=1 | X)
        p1 = gp.predict(X)[0].ravel()
        y_pred = (p1 >= 0.5).astype(int)

        return float(np.mean(y_pred != y_true))





    # def Update(
    #     self,
    #     x, y,
    #     optimize=False,
    #     kern_variance_fix=False,
    #     kern_lengthscale_fix=False,
    #     mean_fix=False,
    #     ll_tol=0,
    #     p_infty=0.02,   # target far-field p(y=1); choose what you want (e.g. 0.02 favours class 0)
    # ):
    #     """
    #     Update model with new data (x,y). Optionally optimise kernel hyperparameters.
    #     After optimisation, update the constant prior mean based on the optimised kernel variance:
    #         m = Phi^{-1}(p_infty) * sqrt(1 + sigma_f^2)
    #     where sigma_f^2 is the kernel variance.
    #     """
    
    #     # -------------------- append data --------------------
    #     x = x.reshape(-1, self.f_num)
    #     X = np.concatenate((self.gpc.X, x), axis=0)
    #     Y = np.concatenate((self.gpc.Y, [[y]]), axis=0)
    
    #     # -------------------- helper: build GP model --------------------
    #     def _build_gp(X, Y, kernel, mean_value):
    #         if mean_value is None:
    #             mean = None
    #         else:
    #             mean = GPy.mappings.Constant(input_dim=self.f_num, output_dim=1, value=float(mean_value))
    #             if mean_fix:
    #                 mean.fix()
    #             else:
    #                 mean.unfix()
    
    #         m = GPy.core.GP(
    #             X=X,
    #             Y=Y,
    #             kernel=kernel,
    #             mean_function=mean,
    #             inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
    #             likelihood=lik
    #         )
    #         return m
    
    #     # -------------------- kernel constraints --------------------
    #     kernel = self.gpc.kern.copy()
    
    #     if vinterval is not None:
    #         kernel.variance.constrain_bounded(vinterval[0], vinterval[1])
    #     if kern_variance_fix:
    #         kernel.variance.fix()
    #     else:
    #         kernel.variance.unfix()
    
    #     if linterval is not None:
    #         kernel.lengthscale.constrain_bounded(linterval[0], linterval[1])
    #     if kern_lengthscale_fix:
    #         kernel.lengthscale.fix()
    #     else:
    #         kernel.lengthscale.unfix()
    
    #     # -------------------- choose current mean value --------------------
    #     # We keep a dynamic mean that can be updated after optimisation.
    #     # Initialise from existing GP mean if present; otherwise from global prior_mean; otherwise None.
    #     if hasattr(self, "prior_mean_dynamic"):
    #         mean_value = self.prior_mean_dynamic
    #     else:
    #         if self.gpc.mean_function is not None:
    #             # GPy Constant mapping stores value in .C
    #             mean_value = float(np.asarray(self.gpc.mean_function.C))
    #         else:
    #             mean_value = prior_mean if prior_mean is not None else None
    #         self.prior_mean_dynamic = mean_value
    
    #     # -------------------- base model (no optimisation yet) --------------------
    #     m_base = _build_gp(X, Y, kernel, mean_value)
    #     ll_base = float(m_base.log_likelihood())
    #     print("Log_likelihood before optimization:", ll_base)
    
    #     best_model = m_base
    #     best_ll = ll_base
    #     best_kernel = kernel
    
    #     # -------------------- optional optimisation --------------------
    #     if optimize:
    #         kernel_opt = kernel.copy()
    #         m_opt = _build_gp(X, Y, kernel_opt, mean_value)
    
    #         m_opt.optimize_restarts(
    #             optimizer="bfgs",
    #             num_restarts=6,
    #             max_iters=500,
    #             verbose=False
    #         )
    
    #         # Rebuild after optimisation (keeps the optimised parameters cleanly in the model)
    #         m_opt = _build_gp(X, Y, kernel_opt, mean_value)
    #         ll_opt = float(m_opt.log_likelihood())
    #         print("Log_likelihood after optimization:", ll_opt)
    
    #         ll_frac = -1.0 * ((ll_opt - ll_base) / ll_base)
    #         print("Fraction of improvement of lml:", ll_frac)
    
    #         if ll_frac > ll_tol:
    #             best_model = m_opt
    #             best_ll = ll_opt
    #             best_kernel = kernel_opt
    #             print("Kernel Hyperparameters are Optimized")
    
    #     # -------------------- UPDATE MEAN AFTER (POSSIBLE) OPTIMISATION --------------------
    #     # If you want to only update mean when variance is not fixed, respect that:
    #     if not kern_variance_fix:
    #         sigma_f2 = float(best_kernel.variance.item())  # kernel variance = sigma_f^2
    
    #         # Phi^{-1}(p_infty) using scipy.stats.norm.ppf
    #         z = float(norm.ppf(p_infty))
    
    #         # New constant mean to preserve far-field class bias under probit prediction
    #         new_mean_value = z * math.sqrt(1.0 + sigma_f2)
    
    #         # Store for future iterations
    #         self.prior_mean_dynamic = new_mean_value
    
    #         # Rebuild the GP with same best kernel but updated mean
    #         best_model = _build_gp(X, Y, best_kernel, new_mean_value)
    #         best_ll = float(best_model.log_likelihood())
    
    #         print(f"Updated prior mean based on kernel variance: {new_mean_value:.6f}")
    #         print(f"Log_likelihood after mean update: {best_ll}")
    
    #     # -------------------- assign updated GP --------------------
    #     self.gpc = best_model


    
    def Update(self, x, y, optimize = False, kern_variance_fix=False, kern_lengthscale_fix=False, mean_fix=False, ll_tol = 0):###########
        x = x.reshape(-1, self.f_num)
        X = np.concatenate((self.gpc.X, x))
        Y = np.concatenate((self.gpc.Y, [[y]]), axis = 0)

        #lik = self.parameters[3]

        # --------------------------------- Change ----------------------------------------------
        kernel = self.gpc.kern

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
            # mean = GPy.mappings.Constant(input_dim=f_num, output_dim=1, value=prior_mean)
            mean = self.gpc.mean_function[0]
            mean = GPy.mappings.Constant(input_dim=f_num, output_dim=1, value=mean)
            if mean_fix:
                mean.fix()   # keep the prior mean fixed during optimisation
            else:
                mean.unfix()
        else:
            mean = None
        # --------------------------------- Change ----------------------------------------------
        
        m = GPy.core.GP(X = X,
                        Y = Y,
                        kernel = kernel,
                        mean_function=mean,         # --------------------------------- Change ---------------------------------------------- 
                        inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                        likelihood = lik)

        # --------------------------------------------------------------------- Change ------------------------------------------------------------------#
        ll_base = float(m.log_likelihood())
        print("Log_likelihood before optimization:", ll_base)
        
        best_model = m
        best_ll = ll_base

        if optimize:
            kernel_opt = kernel.copy()
            print(kernel_opt)
            # kernel_opt.variance.fix()
            m_opt = GPy.core.GP(
                X=X,
                Y=Y,
                kernel=kernel_opt,
                mean_function=mean,
                inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                likelihood=lik,
            )
            
            m_opt.optimize_restarts(optimizer="bfgs", num_restarts=6, max_iters=500, verbose=False)
                        
            print(m_opt.kern.variance)
            print(m_opt.kern.lengthscale)
            m_opt = GPy.core.GP(
                X=X,
                Y=Y,
                kernel=kernel_opt,
                mean_function=mean,
                inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                likelihood=lik,
            )
            ll_opt = float(m_opt.log_likelihood())
            print("Log_likelihood after optimization:", ll_opt)
            
            ll_frac = -1 * ((ll_opt - ll_base) / ll_base)
            print("Fraction of improvement of lml:", ll_frac)
            if (ll_frac > ll_tol):
                best_model = m_opt
                best_ll = ll_opt
                print("Kernel Hyperparameters are Optimized")

        self.gpc = best_model

    
        # if optimize:
        #     m.optimize_restarts(optimizer='bfgs', num_restarts = 40, max_iters=2000, verbose = False)
        #     m = GPy.core.GP(X = X,
        #                     Y = Y,
        #                     kernel = kernel,
        #                     mean_function=mean,         # --------------------------------- Change ---------------------------------------------- 
        #                     inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
        #                     likelihood = lik)
            
        # self.gpc = m


        
        # --------------------------------------------------------------------- Change ------------------------------------------------------------------#

        
        
        # else:
        #     pass
        # if m.kern.lengthscale.item()>4:
        #     kernel = GPy.kern.RBF(f_num, variance = m.kern.variance.item(), lengthscale = 4)

        #     # --------------------------------- Change ----------------------------------------------
        #     mean = GPy.mappings.Constant(input_dim=self.f_num, output_dim=1, value=prior_mean)
        #     mean.fix()
        #     # --------------------------------- Change ----------------------------------------------
            
        #     m = GPy.core.GP(X = X,
        #                 Y = Y,
        #                 kernel = kernel,
        #                 mean_function=mean,         # --------------------------------- Change ---------------------------------------------- #
        #                 inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
        #                 likelihood = lik)

    def XspaceGenerate(self, x_num):
        xspace = XspaceGenerate_(x_num)
        return xspace

    def ObcClassifierError(self, x_num):
        xspace = self.XspaceGenerate(x_num)
        pyTheta = self.predict_proba(xspace)
        yhat = np.argmax(pyTheta, axis = 1)
        py  = GroundTruthProbability(xspace)
        classifier_error = 1-py[np.arange(x_num), yhat]
        # classifier_error = 0
        # # pyTheta = self.predict_proba(xspace)
        # # yhat = np.argmax(pyTheta, axis = 0)
        # # yhat = yhat.astype(int)
        # # pyhat_r = 
        # pyTheta = self.predict_proba(xspace)
        # yhat = np.argmax(pyTheta, axis = 1)

        # for i, _ in enumerate(xspace):
        #     x = xspace[i:i+1]# all the inputs should take 2d array 
        #     pyTheta = self.predict_proba(x)
        #     yhat = np.argmax(pyTheta)######################
        #     yhat = yhat.astype(int)
        #     py  = GroundTruthProbability(x)
        #     classifier_error += (1 - py[yhat])/x_num
        return classifier_error.mean()

    
    # def ModelDraw(self, name):
    #     print(self.gpc.kern)
    #     if f_num >2:
    #         self.gpc.plot(visible_dims = [0])
    #         self.gpc.plot_f(visible_dims = [0])
    #     else:
    #         self.gpc.plot()
    #         plt.savefig('c_'+name)
            
    #         if f_num == 1:
    #             xspace = self.XspaceGenerate(1000)
    #             pspace = np.zeros(len(xspace))
    #             for i, x in enumerate(xspace):
    #                 pspace[i] = GroundTruthProbability(x)[1]
    #             plt.plot(xspace, pspace, 'ro')
    #         #self.gpc.plot_f()
        
        
        #plt.plot(xspace, fspace, 'bo')
            #plt.savefig('f_'+name)
    
    def XspaceGenerateApprox(self, x_num, x):
        # xspace = XspaceGenerateApprox_(x_num, x)
        # return xspace

        # --------------------------------------------------------- Change ---------------------------------------------------------------------------------------
        # d = self.gpc.kern.lengthscale.item()
        # d = 0.2

        
        d = 0.5
        
        # ls = np.asarray(self.gpc.kern.lengthscale)


        # if ls.size == 1:
        #     d = np.full(self.f_num, float(ls))
        # else:
        #     d = ls.reshape(-1)   # shape (f_num,)

        x = np.asarray(x, dtype=float).reshape(1, self.f_num)
        
        xspace = norm.rvs(size=(x_num, self.f_num), loc=x, scale=d)
        xspace, _ = Xtruncated(xinterval[0], xinterval[1], xspace)

        if len(xspace) == 0:
            xspace = np.random.uniform(xinterval[0], xinterval[1], (x_num, self.f_num))
            wspace_log = np.zeros(x_num) + px_log
            return xspace, wspace_log, px_log

        wspace_log_array = norm.logpdf(xspace, loc=x, scale=d)  # (n_points, f_num)
        wspace_log = np.sum(wspace_log_array, axis=1) + np.log(x_num / len(xspace))
    
        return xspace, wspace_log, px_log

# ------------------------------------------------------------ CHANGE -----------------------------------------------------#
    def predict_latent_moments(self, X):
        """
        Return posterior latent mean and marginal variance of f(x) at X.
    
        Parameters
        ----------
        X : array-like, shape (N, d)
            Inputs in normalised space.
    
        Returns
        -------
        mu : np.ndarray, shape (N,)
            Latent posterior mean.
        var : np.ndarray, shape (N,)
            Latent posterior marginal variance (diag of posterior cov).
        """
        X = np.asarray(X, float).reshape(-1, self.f_num)
        xt = torch.tensor(X, dtype=torch.float64)
    
        with torch.no_grad():
            mu_t, Sigma_tt = self._noiseless_predict_torch(xt)  # mu: (N,1), Sigma: (N,N)
    
        mu = mu_t.detach().cpu().numpy().reshape(-1)
        var = torch.diag(Sigma_tt).detach().cpu().numpy().reshape(-1)
        return mu, var
# ------------------------------------------------------------ CHANGE -----------------------------------------------------#


        
        # #xspace = np.random.multivariate_normal(mean = x, scale = d, x_num)
        # if self.f_num > 1:
        #     cov = np.eye(self.f_num)*d
        #     mean = x.reshape(-1)
        #     xspace = np.random.multivariate_normal(mean = mean, cov = cov, size = x_num)
        # else:
        #     xspace = np.random.normal(x, d, (x_num, 1))##this is only for single dimension
        # # idxarray = np.all(xspace >= xinterval[0], axis=1) & np.all(xspace <= xinterval[1], axis = 1)
        # # xspace = xspace[idxarray].reshape(-1, f_num)
        # xspace = norm.rvs(size = (x_num, f_num), loc = x, scale = d)
        # xspace, _ = Xtruncated(xinterval[0], xinterval[1], xspace)

        # # ---------------Self made change---------------------------------------------------------------------------
        # if len(xspace) == 0:
        #     # fallback: simple uniform sampling over the domain
        #     xspace = np.random.uniform(xinterval[0], xinterval[1], (x_num, self.f_num))
        #     # for importance weights we can just set wspace_log constant
        #     wspace_log = np.zeros(x_num) + px_log
        #     return xspace, wspace_log, px_log
        # # ----------------------------------------------------------------------------------------------------------
        
        # wspace_log_array = norm.logpdf(xspace, loc=x, scale=d)
        # wspace_log = np.sum(wspace_log_array, axis=1)+np.log(x_num/len(xspace))
        
        
        # # if f_num>1:
        # #     wspace = multivariate_normal.pdf(xspace, mean = mean, cov = cov )
        # # else:
        # #     wspace = norm.pdf(xspace, loc=x, scale=d)
        # #xspace = np.random.uniform(max(xinterval[0], x-3*d), min(xinterval[1], x+3*d), (x_num, f_num))

        # return xspace, wspace_log, px_log
        # --------------------------------------------------------- Change ---------------------------------------------------------------------------------------

# %%
"""
Implementation note / attribution
--------------------------------
This code is based on the code provided in and from:

Zhao, G., Dougherty, E. R., Yoon, B.-J., Alexander, F. J., & Qian, X. (2021).
*Efficient Active Learning for Gaussian Process Classification by Error Reduction*.

Any deviations from the paper are implementation decisions made for this project.

"""

import numpy as np
from numpy.random import choice
from matplotlib import pyplot as plt
import GPy
from scipy.stats import norm, multivariate_normal
from scipy.stats import truncnorm

f_num = 2 #feature number          #only change here for high-dimension
c_num = 2


# --------------------------------------------- Kernel choice ---------------------------------------------------------- # 
kernel = GPy.kern.RBF(f_num,
                      lengthscale = 0.4,
                      variance=3,
                      ARD=True,
                     )
# kernel = GPy.kern.RBF(f_num, variance = 1, lengthscale = 0.4)

# kernel = GPy.kern.Matern32(
#     input_dim=f_num,
#     variance=1.0,
#     lengthscale=0.4,
#     # ARD=True,
# )

# kernel = GPy.kern.StdPeriodic(
#     input_dim=f_num,
#     variance=1.0,
#     lengthscale=0.4,
#     period=1.0
# )

linterval = None #[0.1, 1]     #[0.05, 4] #length scale bounds, list or None               # ----------------------------------Change --------------------
vinterval = None #[0.99, 1.01]                  #[0.1, 200] # Variance (sigma^2) scale bounds, list or None            # ------------------ Change ---------------

# kernel.lengthscale.constrain_bounded(linterval[0], linterval[1])
# kernel.variance.constrain_bounded(vinterval[0], vinterval[1])

# --------------------------------------------- Kernel choice ---------------------------------------------------------- # 


prior_mean = -2  # ----------------------------------Change -----------------      Negative sets prior to class 0, positive sets prior to class 1 

lik = GPy.likelihoods.Bernoulli()
parameters_ =  [f_num, c_num, kernel, lik]

xinterval = [np.array(0), np.array(1)]

px = 1/(xinterval[1] - xinterval[0])**f_num
px_log = -f_num*np.log(xinterval[1] - xinterval[0])

discrete_label = False
optimize_label = True

Perror = 0.2


def SetGlobalNumber(x_num = 1000): #############################################################################
    return
    


def GroundTruthProbability(x):########################################################################
# x can be single point and xspace

    x = x.reshape(-1, f_num)

    if len(x) == 1:
        x1 = int(x[0, 0])
        x2 = int(x[0, 1])
        py_1 = np.array([[(x1+x2)%2*(1-Perror)+(1-(x1+x2)%2)*Perror
        ]])
    else:
        x1 = x[:, 0:1].astype(int)
        x2 = x[:, 1:2].astype(int)
        py_1 = (x1+x2)%2*(1-Perror)+(1-(x1+x2)%2)*Perror#py_1 should be size xnum*1


    #py_1 = lik.gp_link.transf(mgpr.predict(x)[0])
    # gpm.predict(x)
    
    # j = np.where(np.all(np.isclose(xspace, x), axis = 1))[0][0]
    # py_1 = pspace[j]
    py_0 = 1 - py_1
    #py = np.array([py_0, py_1]) #pymat size x_num*cnum
    py = np.concatenate((py_0, py_1), axis = 1)
    return py

def BayesianError(x_num):
    # x = x.reshape(-1, self.f_num)
    #     py_1 = self.gpc.predict(x)[0]
    #     py_0 = 1 - py_1
    #     pymat = np.concatenate((py_0, py_1), axis = 1) #pymat size x_num*cnum

    return Perror

def ModelDraw(model, name):
    #this is for f_num == 2
    pass
    print(model.gpc.kern)
    model.gpc.plot()
    plt.savefig('c_'+name)

if __name__ == "__main__":
    xspace = np.random.uniform(xinterval[0], xinterval[1], (1, f_num))
    print(GroundTruthProbability(xspace))
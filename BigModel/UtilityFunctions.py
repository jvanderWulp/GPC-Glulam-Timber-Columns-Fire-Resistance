"""
Implementation note / attribution
--------------------------------
This code is based on the code provided in and from:

Zhao, G., Dougherty, E. R., Yoon, B.-J., Alexander, F. J., & Qian, X. (2021).
*Efficient Active Learning for Gaussian Process Classification by Error Reduction*.

Any deviations from the paper are implementation decisions made for this project.

"""


import numpy as np
from scipy import special
from scipy.special import softmax
import torch


def SetGlobal(k_, softtype_, x_num_, approx_label_):
    global k, softtype, x_num, approx_label
    x_num = x_num_
    k = k_
    softtype = softtype_
    approx_label = approx_label_


def SMOCU(x, model, xspace):
    pymat = model.predict_proba(xspace)

    if softtype == 0:  # original MOCU
        obc_correct = np.amax(pymat, axis=1)
        smocu = np.mean(obc_correct)

    if softtype == 1:  # soft MOCU with softmax
        obc_correct = np.sum(softmax(pymat * k, axis=1) * pymat, axis=1)
        smocu = np.mean(obc_correct)

    elif softtype == 2:  # soft MOCU with logsumexp
        obc_correct = special.logsumexp(k * pymat, axis=1) / k
        smocu = np.mean(obc_correct)

    return smocu


def D_SMOCU(x, model):
    smocu2 = 0
    py_x = model.predict_proba(x)
    xspace = model.XspaceGenerate(x_num)
    smocu1 = SMOCU(x, model, xspace)

    for i in range(model.c_num):
        p = py_x.flat[i]
        y = i
        model2 = model.UpdateNew(x, y)
        smocu2 += p * SMOCU(x, model2, xspace)

    return smocu2 - smocu1


def U_SMOCU_K(x, model, k, softtype, x_num, approx_label, version):
    SetGlobal(k, softtype, x_num, approx_label)
    if approx_label is True:
        return D_SMOCU_Approx(x, model, version)
    return D_SMOCU(x, model)


def U_SMOCU(softtype=0, k=1, x_num=1000, approx_label=False):
    return lambda x, model, version='numpy': U_SMOCU_K(x, model, k, softtype, x_num, approx_label, version)


def D_SMOCU_Approx(x, model, version):
    if version == 'pytorch':
        assert (
            type(x) == torch.Tensor and x.requires_grad
        ), "for pytorch, x should be a tensor with required grad"

    x = x.reshape(-1, model.f_num)
    smocu2 = 0

    if version == 'pytorch':

        logsumexp = torch.logsumexp
        mean = torch.mean
        amax = torch.amax
        sum = torch.sum
        softmax_t = torch.softmax  # avoid shadowing scipy softmax
        exp = torch.exp

        py_x = model.predict_proba_torch(x)
        xspace_mean = x.detach().numpy()

    else:

        logsumexp = special.logsumexp
        mean = np.mean
        amax = np.amax
        sum = np.sum
        exp = np.exp

        py_x = model.predict_proba(x)
        xspace_mean = x

    if hasattr(model, 'is_real_data') and model.is_real_data:
        xspace = model.XspaceGenerate(x_num)
        PoverW = np.ones(x_num)
        assert (version == 'numpy')  # in discrete, version is numpy
    else:
        xspace, wspace_log, px_log = model.XspaceGenerateApprox(x_num, xspace_mean)
        wspace_log = wspace_log.reshape(-1)
        PoverW = np.exp(px_log - wspace_log)

    pymat1 = model.predict_proba(xspace)

    if version == 'pytorch':
        PoverW = torch.tensor(PoverW)
        pymat1 = torch.tensor(pymat1)

    def SMOCUApprox(pymat):
        if softtype == 0:  # original MOCU
            obc_correct = amax(pymat, axis=1)

        if softtype == 1:  # soft MOCU with softmax
            if version == 'pytorch':
                obc_correct = sum(softmax_t(pymat * k, dim=1) * pymat, dim=1)
            else:
                obc_correct = np.sum(softmax(pymat * k, axis=1) * pymat, axis=1)

        elif softtype == 2:  # soft MOCU with logsumexp
            obc_correct = logsumexp(k * pymat, axis=1) / k

        smocu = mean(obc_correct * PoverW)
        return smocu

    smocu1 = SMOCUApprox(pymat1)

    pymat20, pymat21 = model.OneStepPredict(x, xspace, version)

    smocu2 = (SMOCUApprox(pymat20) * py_x[0, 0] +
              SMOCUApprox(pymat21) * py_x[0, 1])

    return smocu2 - smocu1

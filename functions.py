import numpy as np
import random
import time

from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from scipy.special import expit
from copy import deepcopy
from scipy.stats import bernoulli
import math
from itertools import permutations
from scipy.spatial.distance import cdist, euclidean

def randk(x, args):
    k = args[0]
    n = len(x)
    indices = np.random.choice(np.arange(n), replace=False, size=(n - k))
    x[indices] = 0
    return x*(len(x)/k), 64 * k

def quantization(x, args):
    n = len(x)
    ord_norm = args[0]
    weight_sum = norm(x, ord=ord_norm)
    xis = bernoulli.rvs(np.true_divide(np.abs(x), weight_sum))
    res = weight_sum * np.sign(x) * xis
    return res, 64 + 2*np.count_nonzero(xis)

def identical(x, args):
    n = len(x)
    return x, 64*n


def topK(v, args):
    k = args[0]
    d = v.shape[0]
    if k == d:
        return v
    if k == 0:
        return np.zeros_like(v)

    v_with_indices = list(zip(v.tolist(), range(d)))
    v_with_indices.sort(key=lambda x: abs(x[0]))

    for i in range(len(v_with_indices) - k):
        index = v_with_indices[i][1]
        v_with_indices[i] = (0, index)

    v_with_indices.sort(key=lambda x: x[1])
    return np.asarray([x[0] for x in v_with_indices]), 64*k

def topk(x, args):
    k = args[0]
    k_order_statistics = np.sort(np.abs(x))[-k]
    return np.where(np.abs(x) >= k_order_statistics, x, 0), 64*k

def logreg_loss(x, args):
    A = args[0]
    y = args[1]
    l2 = args[2]
    setting = args[3]
    assert l2 >= 0
    assert len(y) == A.shape[0]
    assert len(x) == A.shape[1]
    degree1 = np.zeros(A.shape[0])
    degree2 = -A.dot(x) * y
    l = np.logaddexp(degree1, degree2)
    m = y.shape[0]
    if setting == 'PL':
        return np.sum(l) / m + l2/2 * norm(x) ** 2
    else:
        return np.sum(l) / m + l2/2 * (x**2/(1+x**2)).sum()

def logreg_grad(x, args):
    A = args[0]
    y = args[1]
    mu = args[2] 
    setting = args[3]
    assert mu >= 0
    assert len(y) == A.shape[0]
    assert len(x) == A.shape[1]
    degree = -y * (A.dot(x))
    sigmas = expit(degree)
    loss_grad = - A.T.dot(y * sigmas) / A.shape[0]
    assert len(loss_grad) == len(x)
    if setting == 'PL':
        return loss_grad + mu * x 
    else:
        return loss_grad + mu * (x/((1+x**2)**2))

def logreg_grad_plus_lasso(x, args):
    return logreg_grad(x, args) + args[4] * np.sign(x)

def r(x, l1):
    assert (l1 >= 0)
    return l1 * norm(x, ord = 1)

def F(x, args):
    return logreg_loss(x, args) + r(x, args[4])
    
def prox_R(x, l1):
    res = np.abs(x) - l1 * np.ones(len(x))
    return np.where(res > 0, res, 0) * np.sign(x)

def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1

def CM(W, s , G):
    N = math.ceil(len(W)/s)
    z = np.zeros((N, len(G[0])))
    for i in range(N):
        for k in range(i*s, min(len(W), (i+1)*s)):
            z[i] += G[W[k]-1]
    return np.median(z/s, axis=0)

def GM(W, s, G):
    N = math.ceil(len(W)/s)
    z = np.zeros((N, len(G[0])))
    for i in range(N):
        for k in range(i*s, min(len(W), (i+1)*s)):
            z[i] += G[W[k]-1]
    return geometric_median(z/s)
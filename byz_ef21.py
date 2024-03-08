import numpy as np
import random
import time
import pickle
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from scipy.stats import norm as norm_d
from scipy.stats import randint
from scipy.stats import bernoulli
import scipy
from functions import *
from utils import *
from copy import deepcopy
import math
from itertools import permutations
from scipy.spatial.distance import cdist, euclidean

def stopping_criterion(num_of_bits_passed_up, T):
    return (num_of_bits_passed_up <= T)

def byz_ef21(filename, x_init, A, y, clients_A, clients_y, gamma, num_of_byz, num_of_workers,
            attack, agg, sparsificator, sparsificator_params, setting, l2=0, T=5000, max_t=np.inf,
            save_info_period=100, x_star=None, f_star=None):
    
    # m - total number of datasamples
    # n - dimension of the problem
    m, n = A.shape
    assert (len(x_init) == n)
    assert (len(y) == m)

    # Distribute the data
    G = num_of_workers - num_of_byz
    mul = clients_A[0].shape[0]
    A = A[:mul*G]
    y = y[:mul*G]

    if x_star is None:
        x_star = np.zeros(n)
    if f_star is None:
        f_star = 0
    ref_point = np.array(x_star)
    x = np.array(x_init)
    

    G_w = np.zeros((num_of_workers, len(x)))
    G_w1 = logreg_grad(x, [A, y, l2, setting])
    w = np.zeros((num_of_byz, n))
    w_j_max = np.zeros(n)
    w_j_min = np.zeros(n)
    s = np.zeros((num_of_byz, n))
    p1 = 0.5*np.ones((num_of_byz, n))
    p1w = 0.5*np.ones((num_of_byz, n))
    p1h = 0.5*np.ones((num_of_byz, n))
    q = 0.5*np.ones((num_of_byz, n))
    q1 = 0.5*np.ones((num_of_byz, n))

    for i in range(num_of_workers):
        G_w[i] = logreg_grad(x, [clients_A[i], clients_y[i], l2, setting])

    grad = np.mean(G_w, axis=0)

    # Test the distribution of data
    if norm(G_w1 - sum([logreg_grad(x, [clients_A[i], clients_y[i], l2, setting]) for i in range(num_of_workers)]) * 1/num_of_workers) < 1e-10:
        print('Data distributed correctly')
    else:
        print('Problem with Data distribution')

    # Statistics
    it = 0
    its = np.array([0])
    tim = np.array([0.0])
    data_passes = np.array([0.0])
    bits_passes = np.array([0.0])
    func_val = np.array([logreg_loss(x, [A, y, l2, setting]) - f_star])
    sq_distances = np.array([norm(x - ref_point) ** 2])
    norm_grad = np.array([np.linalg.norm(x=grad, ord=2)**2])
    t_start = time.time() # time
    num_of_data_passes = 0.0 # computational complexity
    num_of_bits_passed = 0.0 # communication complexity

    # Method
    while stopping_criterion(num_of_bits_passed, T):

        num_of_bits_passed += sparsificator_params[0]
        num_of_data_passes += 1.0

        x = x - gamma * G_w1
      
        grads = np.zeros((num_of_workers, len(x)))

        for i in range(G):
            A_i = clients_A[i]
            y_i = clients_y[i]
            new_grad = logreg_grad(deepcopy(x), [A_i, y_i, l2, setting])
            grads[i] = new_grad
            G_w[i] += sparsificator(new_grad - G_w[i], sparsificator_params)[0]

        grad = np.mean(grads, axis=0)

        # Attack
        if attack == "BF":
            for i in range(G, num_of_workers):
                A_i = clients_A[i]
                y_i = clients_y[i]
                new_grad = logreg_grad(deepcopy(x), [A_i, y_i, l2, setting])
                G_w[i] -= sparsificator(new_grad - G_w[i], sparsificator_params)[0]
            
        if attack == "LF":
            for i in range(G, num_of_workers):
                A_i = clients_A[i]
                y_i = -1*clients_y[i]
                new_grad = logreg_grad(deepcopy(x), [A_i, y_i, l2, setting])
                G_w[i] += sparsificator(new_grad - G_w[i], sparsificator_params)[0]

            
        if attack == "IPM":
            sum_of_good = sum(G_w[:G] - G_w1)
            for i in range(G, num_of_workers):
                G_w[i] = -0.1 * sum_of_good / (G) + G_w1
                
        if attack == "ALIE":  
            exp_of_good = sum(G_w[:G] - G_w1) / (G)
            var_of_good = (sum((G_w[:G]-G_w1) * (G_w[:G] - G_w1))) / (G) - exp_of_good * exp_of_good
            for i in range(G, num_of_workers):
                G_w[i] = exp_of_good - 1.06 * var_of_good + G_w1
                
        if attack == "LMP":
            for i in range(num_of_workers - num_of_byz, num_of_workers):
                A_i = clients_A[i]
                y_i = clients_y[i]
                new_grad = logreg_grad(deepcopy(x), [A_i, y_i, l2, setting])
                grads[i] = new_grad
                w[i-(num_of_workers - num_of_byz)] += sparsificator(new_grad - G_w[i], sparsificator_params)[0]
                for j in range(n):
                    if agg == "GM":
                        perm = np.random.permutation(num_of_workers-num_of_byz)
                        G_w1 = GM(perm, 2, G_w)
                    if agg == "CM":
                        perm = np.random.permutation(num_of_workers-num_of_byz)
                        G_w1 = CM(perm, 2, G_w)
                    x_k = x - gamma * G_w1
                    s[i-(num_of_workers - num_of_byz)][j] = np.sign(G_w1[j])
                for j in range(n):
                    w_j_max[j] = max(w[i][j] for i in range(num_of_byz))
                for j in range(n):
                    w_j_min[j] = min(w[i][j] for i in range(num_of_byz))
                for j in range(n):
                    if s[i-(num_of_workers - num_of_byz)][j] == -1:
                        if w_j_max[j] >= 0:
                            G_w[i][j] = random.uniform(w_j_max[j], 2 * w_j_max[j])
                        else:
                            G_w[i][j] = random.uniform(w_j_max[j], 0.5 * w_j_max[j])
                    else:
                        if w_j_min[j] >= 0:
                            G_w[i][j] = random.uniform(0.5 * w_j_max[j], w_j_max[j])
                        else:
                            G_w[i][j] = random.uniform(2 * w_j_max[j], w_j_max[j])
                    
                            
        if attack == "ROP":
            for i in range(num_of_workers - num_of_byz, num_of_workers):
                c = i-(num_of_workers - num_of_byz)
                p1[c] = np.ones(n)
                q[c] = G_w1
                if it == 0:
                    q1[c] = G_w1
                q[c] = 0.9 * q1[c] + 0.1 * q[c]
                if norm(q[c]) == 0:
                    G_w[i] = np.zeros(n)
                else:
                    p1w[c] = q[c] * (np.dot(p1[c], q[c]) / np.dot(q[c], q[c]))
                    p1h[c] = p1[c] - p1w[c]
                    dt = p1h[c] / norm(p1h[c])
                    G_w[i] = dt + q1[c]
                    q1[c] = q[c]


        # Aggregation    
        if agg == "GM":
            perm = np.random.permutation(num_of_workers)
            G_w1 = GM(perm, 2, G_w)            
        
        if agg == "M":
            perm = np.random.permutation(num_of_workers)
            G_w1 = np.mean(G_w, axis=0)
        
        if agg == "CM":
            perm = np.random.permutation(num_of_workers)
            G_w1 = CM(perm, 2, G_w)
        
        
        # Save statistics every period of time
        if ((it + 1) % save_info_period == 0):
            its = np.append(its, it + 1)
            tim = np.append(tim, time.time() - t_start)
            data_passes = np.append(data_passes, num_of_data_passes)
            bits_passes = np.append(bits_passes, num_of_bits_passed)
            func_val = np.append(func_val, F(x, [A, y, l2, setting, 0]) - f_star)
            sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
            norm_grad = np.append(norm_grad, np.linalg.norm(x=grad, ord=2)**2)
        if tim[-1] > max_t:
            break

        it += 1


    # Stats at the end
    if (it % save_info_period != 0):
        its = np.append(its, it)
        tim = np.append(tim, time.time() - t_start)
        data_passes = np.append(data_passes, num_of_data_passes)
        bits_passes = np.append(bits_passes, num_of_bits_passed)
        func_val = np.append(func_val, logreg_loss(x, [A, y, l2, setting]) - f_star)
        sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
        norm_grad = np.append(norm_grad, np.linalg.norm(x=grad, ord=2)**2)

    # Save the data
    res = {'last_iter': x, 'func_vals': func_val, 'norm_grad': norm_grad, 'iters': its, 'time': tim, 'data_passes': data_passes, 'bits_passes':bits_passes,
           'squared_distances': sq_distances, 'num_of_workers': num_of_workers, 'num_of_byz': num_of_byz}
    
    with open("dump/" + filename + "_Byz_EF21_" + attack + "_" + agg + "_gamma_" + str(gamma) + "_l2_" + str(l2) + "_epochs_" + str(T) + "_workers_" + str(num_of_workers) + "_byz_" + str(num_of_byz)+ "_K_" + str(sparsificator_params[0]) + ".txt", 'wb') as file:
        pickle.dump(res, file)
    return res
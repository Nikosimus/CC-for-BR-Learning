import numpy as np
from numpy.linalg import norm
import pickle
import matplotlib.pyplot as plt
import itertools
from scipy.stats import norm as norm_d
from scipy.stats import expon
from scipy.stats import weibull_min as weibull
from scipy.stats import burr12 as burr
from scipy.stats import randint
from scipy.stats import uniform
from scipy.optimize import minimize
import copy
import math
import time
from scipy.optimize import minimize
from scipy.sparse.linalg import svds
from scipy.linalg import svdvals
import scipy
from sklearn.datasets import load_svmlight_file
import pickle
from pathlib import Path


def prepare_data(dataset):
    filename = "datasets/" + dataset + ".txt"

    data = load_svmlight_file(filename)
    A, y = data[0], data[1]
    m, n = A.shape
    
    if (0 in y) & (1 in y):
        y = 2 * y - 1
    if (2 in y) & (4 in y):
        y = y - 3
    if (1 in y) & (2 in y):
        y = 2 * y - 3 
    assert((-1 in y) & (1 in y))
    
    sparsity_A = A.count_nonzero() / (m * n)
    return A, y, m, n, sparsity_A


def distrib_data(A, y, num_of_workers, num_of_byz):
    m, n = A.shape
    G = num_of_workers - num_of_byz
    mul = int(m/G)
    clients_A = []
    clients_y = []
    for i in range(G):
        clients_A.append(A[mul*i : mul*(i+1)])
        clients_y.append(y[mul*i : mul*(i+1)]) 
    for i in range(num_of_byz):
        clients_A.append(A)
        clients_y.append(y)
    return clients_A, clients_y
    

def compute_L(dataset, A, clients_A, num_of_byz, l2):
    filename = "dump/" + dataset + str(l2) + "_L.txt"
    file_path = Path(filename)
    if file_path.is_file():
        with open(filename, 'rb') as file:
            L, average_L, worst_L = pickle.load(file)
    else:
        num_of_workers = len(clients_A)
        G = num_of_workers - num_of_byz
        m = A.shape[0]
        mul = clients_A[0].shape[0]

        sigmas = svds(A, return_singular_vectors=False)
        L = sigmas.max()**2 / (4*m) + l2

        average_L = 0
        for i in range(G):
            average_L += (svds(clients_A[i], return_singular_vectors=False).max()**2 / (4*mul) + l2)**2/G
        average_L = np.sqrt(average_L)

        worst_L = 0
        denseA = A.toarray()
        for i in range(m):
            L_temp = (norm(denseA[i])**2)*1.0 / 4
            if L_temp > worst_L:
                worst_L = L_temp
        worst_L += l2

        with open(filename, 'wb') as file:
            pickle.dump([L, average_L, worst_L], file)
    return L, average_L, worst_L

def prepare_data_distrib(dataset, data_size, num_of_workers, l2):
    filename = "datasets/" + dataset + str(l2) + ".txt"

    data = load_svmlight_file(filename)
    A, y = data[0], data[1]
    m, n = A.shape
    assert(data_size <= m)
    
    size_of_local_data = int(data_size*1.0 / num_of_workers)
    A = A[0:size_of_local_data*num_of_workers]
    y = y[0:size_of_local_data*num_of_workers]
    m, n = A.shape
    assert(data_size == size_of_local_data*num_of_workers)
    
    perm = np.random.permutation(m)
    data_split = perm[0:size_of_local_data]
    for i in range(num_of_workers-1):
        data_split = np.vstack((data_split, perm[(i+1)*size_of_local_data:(i+2)*size_of_local_data]))
    
    if (0 in y) & (1 in y):
        y = 2 * y - 1
    if (2 in y) & (4 in y):
        y = y - 3
    assert((-1 in y) & (1 in y))
    
    sparsity_A = A.count_nonzero() / (m * n)
    return A, y, m, n, sparsity_A, data_split

def compute_L_distrib(dataset, A):
    filename = "dump/"+dataset+"_"+str(A.shape[0])+"_"+"_L.txt"
    file_path = Path(filename)
    if file_path.is_file():
        with open(filename, 'rb') as file:
            L, average_L, worst_L = pickle.load(file)
    else:
        sigmas = svds(A, return_singular_vectors=False)
        m = A.shape[0]
        L = sigmas.max()**2 / (4*m)
        
        worst_L = 0
        average_L = 0
        denseA = A.toarray()
        for i in range(m):
            L_temp = (norm(denseA[i])**2)*1.0 / 4
            average_L += L_temp / m
            if L_temp > worst_L:
                worst_L = L_temp
        with open(filename, 'wb') as file:
            pickle.dump([L, average_L, worst_L],file)
    return L, average_L, worst_L

def save_split(dataset, num_of_workers, data_split):
    filename = "dump/"+dataset+"_split_num_of_workers_"+str(num_of_workers)+".txt"
    with open(filename, 'wb') as file:
        pickle.dump(data_split, file)
        
def read_split(dataset, num_of_workers):
    filename = "dump/"+dataset+"_split_num_of_workers_"+str(num_of_workers)+".txt"
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_problem(dataset, num_of_workers, params):
    filename = "dump/"+dataset+"_problem_num_of_workers_"+str(num_of_workers)+".txt"
    with open(filename, 'wb') as file:
        pickle.dump(params, file)
        
def read_problem(dataset, num_of_workers):
    filename = "dump/"+dataset+"_problem_num_of_workers_"+str(num_of_workers)+".txt"
    with open(filename, 'rb') as file:
        return pickle.load(file)


def save_solution(dataset, l2, l1, x_star, f_star):
    filename = "dump/"+dataset+"_solution_l2_"+str(l2)+"_l1_"+str(l1)+".txt"
    with open(filename, 'wb') as file:
        pickle.dump([x_star, f_star], file)

def read_solution(dataset, l2, l1):
    with open('dump/'+dataset+'_solution_l2_'+str(l2)+"_l1_"+str(l1)+".txt", 'rb') as file:
        return pickle.load(file)


def read_results_from_file(filename, method, args):
    if method == "Byz_MARINA":
        with open('dump/'+filename+"_Byz_MARINA_No_Sync_"+args[6]+"_"+args[7]+"_gamma_"+str(args[0])+"_l2_"+str(args[1]) +"_p_"+str(args[2])
                  +"_epochs_"+str(args[3])+"_workers_"+str(args[4])+"_byz_"+str(args[5]) + "_K_" + str(args[8])
                  +".txt", 'rb') as file:
            return pickle.load(file)
    if method == "Byz_EF21":
        with open('dump/'+filename+'_Byz_EF21_'+args[5]+"_"+args[6]+"_gamma_"+str(args[0])+"_l2_"+str(args[1])
                  +"_epochs_"+str(args[2])+"_workers_"+str(args[3])+"_byz_"+str(args[4]) + "_K_" + str(args[7])
                  +".txt", 'rb') as file:
            return pickle.load(file)
    if method == "Byz_VR_MARINA":
        with open('dump/'+filename+'_Byz_VR_MARINA_'+args[7]+"_"+args[8]+"_gamma_"+str(args[0])+"_l2_"+str(args[1]) +"_p_"+str(args[2])
                  +"_epochs_"+str(args[3])+"_workers_"+str(args[4])+"_batch_"+str(args[5])+"_byz_"+str(args[6])
                  +".txt", 'rb') as file:
            return pickle.load(file)
    if method == "Byz_DASHA_PAGE":
        with open('dump/'+filename+'_Byz_DASHA_PAGE_'+args[7]+"_"+args[8]+"_gamma_"+str(args[0])+"_l2_"+str(args[1]) +"_p_"+str(args[2])
                  +"_epochs_"+str(args[3])+"_workers_"+str(args[4])+"_batch_"+str(args[5])+"_byz_"+str(args[6])
                  +".txt", 'rb') as file:
            return pickle.load(file)


def make_plots(args):
    supported_modes_y = ['func_vals', 'squared_distances', 'norm_grad']
    supported_modes_x = ['iters', 'data_passes', 'bits_passes', 'time']
    
    filename = args[0]
    mode_y = args[1]
    mode_x = args[2]
    figsize = args[3]
    sizes = args[4]
    title = args[5]
    methods = args[6]
    bbox_to_anchor = args[7]
    legend_loc = args[8]
    mode = args[9]
    save_fig = args[10]
    
    
    title_size = sizes[0]
    linewidth = sizes[1]
    markersize = sizes[2]
    legend_size = sizes[3]
    xlabel_size = sizes[4]
    ylabel_size = sizes[5]
    xticks_size = sizes[6]
    yticks_size = sizes[7]
    
    assert(mode_y in supported_modes_y)
    assert(mode_x in supported_modes_x)
    
    fig = plt.figure(figsize=figsize)
    plt.title(title, fontsize=title_size)
    marker = itertools.cycle(('+', 'd', 'x', 'o', '^', 's', '*', 'p', '<', '>', '^'))
    
    num_of_methods = len(methods)
    for idx, method in enumerate(methods):
        res = read_results_from_file(filename, method[0], method[1])
        print(method)
        print(method[1])
        if method[2] == None:
            length = len(res['iters'])
        else:
            length = method[2]
        if mode == "comparison":
            if method[3]==None:
                plt.semilogy(res[mode_x][0:length], res[mode_y][0:length] / res[mode_y][0], linewidth=linewidth, marker=next(marker), 
                    markersize = markersize, 
                    markevery=range(-idx*int(length/(10*num_of_methods)), len(res[mode_x][0:length]), int(length/10)), 
                    label = method[0])
            else:
                plt.semilogy(res[mode_x][0:length], res[mode_y][0:length] / res[mode_y][0], linewidth=linewidth, marker=next(marker), 
                    markersize = markersize, 
                    markevery=range(-idx*int(length/(10*num_of_methods)), len(res[mode_x][0:length]), int(length/10)), 
                    label = method[3])
        else:
            plt.semilogy(res[mode_x][0:length], res[mode_y][0:length] / res[mode_y][0], linewidth=linewidth, marker=next(marker), 
                markersize = markersize, 
                markevery=range(-idx*int(length/(10*num_of_methods)), len(res[mode_x][0:length]), int(length/10)), 
                label = method[3])
        
    
    plt.legend(bbox_to_anchor=bbox_to_anchor, loc=legend_loc, fontsize=legend_size)

    if mode_x == 'time':
        plt.xlabel(r"Time, $s$", fontsize=xlabel_size)
    if mode_x == 'data_passes':
        plt.xlabel(r"Number of passes through the data", fontsize=xlabel_size)
    if mode_x == 'iters':
        plt.xlabel(r"Number of iterations", fontsize=xlabel_size)
    if mode_x == 'bits_passes':
        plt.xlabel(r"Number of bits sent", fontsize=xlabel_size)

    if mode_y == 'func_vals':
        plt.ylabel(r"$\frac{f(x^k)-f(x^*)}{f(x^0)-f(x^*)}$", fontsize=ylabel_size)
    if mode_y == 'squared_distances':
        plt.ylabel(r"$\frac{||x^k - x^*||_2^2}{||x^0 - x^*||_2^2}$", fontsize=ylabel_size)
    if mode_y == 'norm_grad':
        plt.ylabel(r"$|| \nabla f(x)||^2$", fontsize=ylabel_size)
    plt.xticks(fontsize=xticks_size)
    _ = plt.yticks(fontsize=yticks_size)
    
    ax = fig.gca()
    ax.xaxis.offsetText.set_fontsize(xlabel_size - 2)
    ax.yaxis.offsetText.set_fontsize(ylabel_size - 2)
    
    if save_fig[0]:
        plt.savefig("plot/" + save_fig[1], bbox_inches='tight')
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 00:53:12 2017

@author: kathy
"""

import numpy as np
from scipy.io import  loadmat
import matplotlib.pyplot as plt


def load():
    
    """
        load the data 
    """
    load_data = loadmat('2_data.mat')
    data = load_data['x']
    target = load_data['t']
    
    return data, target

def sigma(a):
    """
        sigmoid function
    """
    sigma = 1/(1+np.exp(-a))
    return sigma

def mu(j, M):
    """
       parameter mu in sigmoid function 
    """
    mu = 2*j/M
    return mu

def design_matrix(data, N, M):
    """
        make the design matrix phi
    """
    x_basis = sigma((data[0:N]-mu(0,M))/s)
    for j in list(range(1, M)):
        x_basis = np.hstack((x_basis, sigma((data[0:N]-mu(j,M))/s)))
    return x_basis

def covar_matrix(S0, Phi):
    """
        calculate the covariance matrix S_N
        SN^-1 = S0^-1 + phi.T*phi
    """
    SN = np.linalg.inv(np.linalg.inv(S0) + np.dot(Phi.T, Phi))
    return SN

def mean_vector(m0, S0, Phi, SN, t):
    """
        calculate the mean vector
        mN = SN(S0^-1 * m0 + phi.T*t)
    """
    mN = np.dot(SN, np.dot(np.linalg.inv(S0), m0) + np.dot(Phi.T, t))
    return mN

def variance(Phi, SN, N):
    """
        var = 1 + phi(x).T*SN*phi(x)
    """
    var = 1 + np.dot(np.dot(Phi[0, :].T, SN), Phi[0, :])
    for n in list(range(1, N)):
        var = np.vstack((var, 1 + np.dot(np.dot(Phi[n, :].T, SN), Phi[n, :])))
    return var

def getCurve(N, w, Phi):
    """
        y(x, w) = w.T*phi(x)
    """
    y = np.dot(w.T, Phi[0, :])
    for n in list(range(1, N)):
        y = np.vstack((y, np.dot(w.T, Phi[n, :])))
    return y

if __name__ == '__main__':
    
    # loading data
    data, target = load()
    
    # innitial setting
    M = 7
    s = 0.1
    N = [10, 15, 30, 80]
    [phi_10, phi_15, phi_30, phi_80] = [design_matrix(data, N[0], M), \
                                        design_matrix(data, N[1], M), \
                                        design_matrix(data, N[2], M), \
                                        design_matrix(data, N[3], M)]
    x = np.linspace(0, 2, 100)
    new_phi = design_matrix(x.reshape(len(x),1), len(x), M)
    
    ###--- problem1 ---###
    # innitial setting
    print ('<---mean vector m_N & covariance matrix S_N--->\n')
    m0 = np.zeros((M,1))
    S0 = np.linalg.inv(np.dot(10**(-6), np.identity(M)))
    # covariance matrix & mean vector
    [S10, S15, S30, S80] = [covar_matrix(S0, phi_10), \
                            covar_matrix(S0, phi_15), \
                            covar_matrix(S0, phi_30), \
                            covar_matrix(S0, phi_80)]
    
    [m10, m15, m30, m80] = [mean_vector(m0, S0, phi_10, S10, target[0:N[0]]), \
                            mean_vector(m0, S0, phi_15, S15, target[0:N[1]]), \
                            mean_vector(m0, S0, phi_30, S30, target[0:N[2]]), \
                            mean_vector(m0, S0, phi_80, S80, target[0:N[3]])]
    # print the result
    print('mean vector m_N = \n')
    print('m10 :\n', m10, '\nm15 :\n', m15, '\nm30 :\n', m30, '\nm80 :\n', m80)
    print('covariance matric S_N = \n')
    print('S10 :\n', S10, 'S15 :\n', S15, '\nS30 :\n', S30, '\nS80 :\n', S80)
    
    ###--- Problem2 ---###
    print ('<--- ﬁve curve samples from the parameter posterior distribution--->\n')
    # print Fig. 3.9
    plt.figure(0)
    plt.plot(data[0:N[0]], target[0:N[0]], 'o', label='target value')
    for i in list(range(5)):
        w10 = np.random.multivariate_normal(m10.reshape(-1), S10)
        sc10 = getCurve(len(x), w10.reshape(7, 1), new_phi)
        plt.plot(x, sc10, 'r--')
    
    plt.figure(1)
    plt.plot(data[0:N[1]], target[0:N[1]], 'o', label='target value')
    for i in list(range(5)):
        w15 = np.random.multivariate_normal(m15.reshape(-1), S15)
        sc15 = getCurve(len(x), w15.reshape(7, 1), new_phi)
        plt.plot(x, sc15, 'r--')
    
    plt.figure(2)
    plt.plot(data[0:N[2]], target[0:N[2]], 'o', label='target value')
    for i in list(range(5)):
        w30 = np.random.multivariate_normal(m30.reshape(-1), S30)
        sc30 = getCurve(len(x), w30.reshape(7, 1), new_phi)
        plt.plot(x, sc30, 'r--')
        
    plt.figure(3)
    plt.plot(data[0:N[3]], target[0:N[3]], 'o', label='target value')
    for i in list(range(5)):
        w80 = np.random.multivariate_normal(m80.reshape(-1), S80)
        sc80 = getCurve(len(x), w80.reshape(7, 1), new_phi)
        plt.plot(x, sc80, 'r--')
    
    ###--- Problem3 ---###
    print ('<---  the predictive distribution of target value --->\n')    
    # mean curve
    [y10, y15, y30, y80] = [getCurve(len(x), m10, new_phi), \
                            getCurve(len(x), m15, new_phi), \
                            getCurve(len(x), m30, new_phi), \
                            getCurve(len(x), m80, new_phi)]
    # variance
    [v10, v15, v30, v80] = [variance(new_phi, S10, len(x)), \
                            variance(new_phi, S15, len(x)), \
                            variance(new_phi, S30, len(x)), \
                            variance(new_phi, S80, len(x))]
    # standard deviation
    [sd10, sd15, sd30, sd80] = [np.sqrt(v10), np.sqrt(v15), np.sqrt(v30), np.sqrt(v80)]
    # region of variance 
    [up10, up15, up30, up80] = [y10 + sd10, y15 + sd15, y30 + sd30, y80 + sd80]
    [low10, low15, low30, low80] = [y10 - sd10, y15 - sd15, y30 - sd30, y80 - sd80]
    # print Fig. 3.8
    plt.figure(4)
    plt.plot(data[0:N[0]], target[0:N[0]], 'o', label='target value')
    plt.plot(x, y10, 'r-')
    plt.fill_between(x, up10.reshape(-1), low10.reshape(-1), color='pink')

    plt.figure(5)
    plt.plot(data[0:N[1]], target[0:N[1]], 'o', label='target value')
    plt.plot(x, y15, 'r-')
    plt.fill_between(x, up15.reshape(-1), low15.reshape(-1), color='pink')

    plt.figure(6)
    plt.plot(data[0:N[2]], target[0:N[2]], 'o', label='target value')
    plt.plot(x, y30, 'r-')
    plt.fill_between(x, up30.reshape(-1), low30.reshape(-1), color='pink')

    plt.figure(7)
    plt.plot(data[0:N[3]], target[0:N[3]], 'o', label='target value')
    plt.plot(x, y80, 'r-')
    plt.fill_between(x, up80.reshape(-1), low80.reshape(-1), color='pink')
    
    
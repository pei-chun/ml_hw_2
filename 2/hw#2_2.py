#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 00:53:12 2017

@author: kathy
"""

import pandas as pd
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

def design_matrix(N, M):
    """
        make the design matrix phi
    """
    data, target = load()
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
    
if __name__ == '__main__':
    
    # loading data
    data, target = load()
    
    # innitial setting
    M = 7
    s = 0.1
    N = [10, 15, 30, 80]
    [phi_10, phi_15, phi_30, phi_80] = [design_matrix(N[0], M), \
                                        design_matrix(N[1], M), \
                                        design_matrix(N[2], M), \
                                        design_matrix(N[3], M)]
    
    ###--- problem1 ---###
    # innitial setting
    print ('<---mean vector m_N & covariance matrix S_N--->\n')
    m0 = np.zeros((M,1))
    S0 = np.linalg.inv(10**(-6)*np.identity(M))
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
    
    
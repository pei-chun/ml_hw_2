#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 00:08:44 2017

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
    train = pd.read_csv("train.csv", sep = ",", header=None) 
    test = pd.read_csv("test.csv", sep = ",", header=None)
    train_t, train_x = train.T[0:3], train.T[3:16]
    train_t, train_x = train_t.T, train_x.T
    return train_t, train_x, test

def activations(w, phi):
    """
        activations
    """
    a = np.dot(w, phi[0].reshape(13, 1))
    for n in list(range(1, 148)):
        a = np.vstack((a, np.dot(w, phi[n].reshape(13, 1))))
    return a

def soft(a1, a2, a3):
    """
        softmax function
    """
    y = 1/(np.exp(a1[0]-a1[0])+np.exp(a2[0]-a1[0])+np.exp(a3[0]-a1[0]))
    for n in list(range(1, 148)):
        y = np.vstack((y, 1/(np.exp(a1[n]-a1[n])+np.exp(a2[n]-a1[n])+np.exp(a3[n]-a1[n]))))
    return y

def gradient(y, k, t, phi):
    """
        gradient
    """
    D = (y[0]-t[0, k-1])*phi[0].reshape(1, 13)
    for n in list(range(1, 148)):
        D = D + (y[n]-t[n, k-1])*phi[n].reshape(1, 13)
    return D

def Hessian(y, phi):
    """
        Hessian matrix
    """
    H = y[0]*(1-y[0])*np.dot(phi[0].reshape(13,1), phi[0].reshape(1, 13))
    for n in list(range(1, 148)):
        H = H + y[n]*(1-y[n])*np.dot(phi[n].reshape(13,1), phi[n].reshape(1, 13))
    return H

def cross(y1, y2, y3, t):
    """
        crossentropy function
    """
    E1, E2, E3 = t[0, 1-1]*np.log(y1[0]), t[0, 2-1]*np.log(y2[0]), t[0, 3-1]*np.log(y3[0])
    for n in list(range(1, 148)):
        E1, E2, E3 = E1 + t[n, 1-1]*np.log(y1[n]), \
                     E2 + t[n, 2-1]*np.log(y2[n]), \
                     E3 + t[n, 3-1]*np.log(y3[n])
    E = E1 + E2 + E3
    return -E

if __name__ == '__main__':
    # loading data
    train_t, train_x, test = load()
    # initial setting
    phi = np.array(train_x)
    target = np.array(train_t)
    w1, w2, w3 = np.zeros((1, 13)), np.zeros((1, 13)), np.zeros((1, 13))
    I = 10**(-6)*np.identity(13)
    
    a1, a2, a3 = activations(w1, phi), activations(w2, phi), activations(w3, phi)
    y1, y2, y3 = soft(a1, a2, a3), soft(a2, a1, a3), soft(a3, a1, a2)
    D1, D2, D3 = gradient(y1, 1, target, phi), gradient(y2, 2, target, phi), gradient(y3, 3, target, phi)
    H1, H2, H3 = Hessian(y1, phi)+I, Hessian(y2, phi)+I, Hessian(y3, phi)+I
    E = cross(y1, y2, y3, target)
    Loss = np.hstack((E))
    #
    
    w1, w2, w3 = w1 - np.dot(D1, np.linalg.inv(H1)), \
                 w2 - np.dot(D2, np.linalg.inv(H2)), \
                 w3 - np.dot(D3, np.linalg.inv(H3))
    a1, a2, a3 = activations(w1, phi), activations(w2, phi), activations(w3, phi)
    y1, y2, y3 = soft(a1, a2, a3), soft(a2, a1, a3), soft(a3, a1, a2)
    
    D1, D2, D3 = gradient(y1, 1, target, phi), gradient(y2, 2, target, phi), gradient(y3, 3, target, phi)
    H1, H2, H3 = Hessian(y1, phi), Hessian(y2, phi), Hessian(y3, phi)
    E = cross(y1, y2, y3, target)
    Loss = np.hstack((Loss, E))
    #
    
    w1, w2, w3 = w1 - np.dot(D1, np.linalg.inv(H1)), \
                 w2 - np.dot(D2, np.linalg.inv(H2)), \
                 w3 - np.dot(D3, np.linalg.inv(H3))
    a1, a2, a3 = activations(w1, phi), activations(w2, phi), activations(w3, phi)
    y1, y2, y3 = soft(a1, a2, a3), soft(a2, a1, a3), soft(a3, a1, a2)
    
    D1, D2, D3 = gradient(y1, 1, target, phi), gradient(y2, 2, target, phi), gradient(y3, 3, target, phi)
    H1, H2, H3 = Hessian(y1, phi), Hessian(y2, phi), Hessian(y3, phi)
    E = cross(y1, y2, y3, target)
    Loss = np.hstack((Loss, E))
    
    #
    w1, w2, w3 = w1 - np.dot(D1, np.linalg.inv(H1)), \
                 w2 - np.dot(D2, np.linalg.inv(H2)), \
                 w3 - np.dot(D3, np.linalg.inv(H3))
    a1, a2, a3 = activations(w1, phi), activations(w2, phi), activations(w3, phi)
    y1, y2, y3 = soft(a1, a2, a3), soft(a2, a1, a3), soft(a3, a1, a2)
    
    D1, D2, D3 = gradient(y1, 1, target, phi), gradient(y2, 2, target, phi), gradient(y3, 3, target, phi)
    H1, H2, H3 = Hessian(y1, phi), Hessian(y2, phi), Hessian(y3, phi)
    E = cross(y1, y2, y3, target)
    Loss = np.hstack((Loss, E))
    #
    w1, w2, w3 = w1 - np.dot(D1, np.linalg.inv(H1)), \
                 w2 - np.dot(D2, np.linalg.inv(H2)), \
                 w3 - np.dot(D3, np.linalg.inv(H3))
    a1, a2, a3 = activations(w1, phi), activations(w2, phi), activations(w3, phi)
    y1, y2, y3 = soft(a1, a2, a3), soft(a2, a1, a3), soft(a3, a1, a2)
    
    D1, D2, D3 = gradient(y1, 1, target, phi), gradient(y2, 2, target, phi), gradient(y3, 3, target, phi)
    H1, H2, H3 = Hessian(y1, phi), Hessian(y2, phi), Hessian(y3, phi)
    E = cross(y1, y2, y3, target)
    Loss = np.hstack((Loss, E))
    
    w1, w2, w3 = w1 - np.dot(D1, np.linalg.inv(H1)), \
                 w2 - np.dot(D2, np.linalg.inv(H2)), \
                 w3 - np.dot(D3, np.linalg.inv(H3))
    a1, a2, a3 = activations(w1, phi), activations(w2, phi), activations(w3, phi)
    y1, y2, y3 = soft(a1, a2, a3), soft(a2, a1, a3), soft(a3, a1, a2)
    
    D1, D2, D3 = gradient(y1, 1, target, phi), gradient(y2, 2, target, phi), gradient(y3, 3, target, phi)
    H1, H2, H3 = Hessian(y1, phi), Hessian(y2, phi), Hessian(y3, phi)
    E = cross(y1, y2, y3, target)
    Loss = np.hstack((Loss, E))
    ###--- problem3-1 ---###
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 00:08:44 2017

@author: kathy
"""

import pandas as pd
import numpy as np
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

def activations(w, phi,t):
    """
        activations
    """
    a = np.dot(w, phi[0].reshape(len(phi.T), 1))
    for n in list(range(1, len(t))):
        a = np.vstack((a, np.dot(w, phi[n].reshape(len(phi.T), 1))))
    return a

def soft(a1, a2, a3, t):
    """
        softmax function
    """
    y = 1/(np.exp(a1[0]-a1[0])+np.exp(a2[0]-a1[0])+np.exp(a3[0]-a1[0]))
    for n in list(range(1, len(t))):
        y = np.vstack((y, 1/(np.exp(a1[n]-a1[n])+np.exp(a2[n]-a1[n])+np.exp(a3[n]-a1[n]))))
    return y

def gradient(y, k, t, phi):
    """
        gradient
    """
    D = (y[0]-t[0, k-1])*phi[0].reshape(1, len(phi.T))
    for n in list(range(1, len(t))):
        D = D + (y[n]-t[n, k-1])*phi[n].reshape(1, len(phi.T))
    return D

def Hessian(y, phi, t):
    """
        Hessian matrix
    """
    H = y[0]*(1-y[0])*np.dot(phi[0].reshape(len(phi.T),1), phi[0].reshape(1, len(phi.T)))
    for n in list(range(1, len(t))):
        H = H + y[n]*(1-y[n])*np.dot(phi[n].reshape(len(phi.T),1), phi[n].reshape(1, len(phi.T)))
    return H

def cross(y1, y2, y3, t):
    """
        crossentropy function
    """
    E1, E2, E3 = t[0, 1-1]*np.log(y1[0]), t[0, 2-1]*np.log(y2[0]), t[0, 3-1]*np.log(y3[0])
    for n in list(range(1, len(t))):
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
    epsilon = 0.001
    a1, a2, a3 = activations(w1, phi, target), activations(w2, phi, target), activations(w3, phi, target)
    y1, y2, y3 = soft(a1, a2, a3, target), soft(a2, a1, a3, target), soft(a3, a1, a2, target)
    D1, D2, D3 = gradient(y1, 1, target, phi), gradient(y2, 2, target, phi), gradient(y3, 3, target, phi)
    H1, H2, H3 = Hessian(y1, phi, target)+I, Hessian(y2, phi, target)+I, Hessian(y3, phi, target)+I
    E = cross(y1, y2, y3, target)
    Loss = np.hstack((E))
    #
    counter = 1
    while epsilon < E:
        w1, w2, w3 = w1 - np.dot(D1, np.linalg.inv(H1)), \
                     w2 - np.dot(D2, np.linalg.inv(H2)), \
                     w3 - np.dot(D3, np.linalg.inv(H3))
        a1, a2, a3 = activations(w1, phi, target), activations(w2, phi, target), activations(w3, phi, target)
        y1, y2, y3 = soft(a1, a2, a3, target), soft(a2, a1, a3, target), soft(a3, a1, a2, target)
        D1, D2, D3 = gradient(y1, 1, target, phi), gradient(y2, 2, target, phi), gradient(y3, 3, target, phi)
        H1, H2, H3 = Hessian(y1, phi, target), Hessian(y2, phi, target), Hessian(y3, phi, target)
        E = cross(y1, y2, y3, target)
        Loss = np.hstack((Loss, E))
        counter = counter + 1
    
    ###--- problem3-1 ---###
    plt.figure(0)
    plt.plot(range(0, counter), Loss)
    
    ###--- problem3-2 ---###
    # test data
    test = np.array(test)
    at1, at2, at3 = activations(w1, test, test), activations(w2, test, test), activations(w3, test, test)
    yt1, yt2, yt3 = soft(at1, at2, at3, test), soft(at2, at1, at3, test), soft(at3, at2, at1, test)
    yt = np.hstack((yt1, yt2, yt3))
    print(np.round(yt))
    
    ##--- problem3-3 ---###
    for n in range(13):
        plt.figure(n+1)
        plt.hist(phi[0:49, n], alpha = 0.5, color='r', label = 'class1')
        plt.hist(phi[49:110, n], alpha = 0.5, color= 'g', label = 'class2')
        plt.hist(phi[110:148, n], alpha = 0.5, color='b', label = 'class3')
        
    ##--- problem3-5 ---###
    contribute = phi[:, 0:2]
    plt.figure(14)
    plt.plot(contribute[0:49,0], contribute[0:49,1], 'o', color='r', label = 'class1')
    plt.plot(contribute[49:110,0], contribute[49:110,1], 'o', color='g', label = 'class2')
    plt.plot(contribute[110:148,0], contribute[110:148,1], 'o', color='b', label = 'class3')
    
    ##--- problem3-6 ---###
    rw1, rw2, rw3 = np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2))
    I = 10**(-6)*np.identity(2)
    epsilon = 0.001
    ra1, ra2, ra3 = activations(rw1, contribute, target), activations(rw2, contribute, target), activations(rw3, contribute, target)
    ry1, ry2, ry3 = soft(ra1, ra2, ra3, target), soft(ra2, ra1, ra3, target), soft(ra3, ra1, ra2, target)
    rD1, rD2, rD3 = gradient(ry1, 1, target, contribute), gradient(ry2, 2, target, contribute), gradient(ry3, 3, target, contribute)
    rH1, rH2, rH3 = Hessian(ry1, contribute, target)+I, Hessian(ry2, contribute, target)+I, Hessian(ry3, contribute, target)+I
    rE = cross(ry1, ry2, ry3, target)
    rLoss = np.hstack((rE))
    #
    counter = 1
    while epsilon < rE:
        rw1, rw2, rw3 = rw1 - np.dot(rD1, np.linalg.inv(rH1)), \
                        rw2 - np.dot(rD2, np.linalg.inv(rH2)), \
                        rw3 - np.dot(rD3, np.linalg.inv(rH3))
        ra1, ra2, ra3 = activations(rw1, contribute, target), activations(rw2, contribute, target), activations(rw3, contribute, target)
        ry1, ry2, ry3 = soft(ra1, ra2, ra3, target), soft(ra2, ra1, ra3, target), soft(ra3, ra1, ra2, target)
        rD1, rD2, rD3 = gradient(ry1, 1, target, contribute), gradient(ry2, 2, target, contribute), gradient(ry3, 3, target, contribute)
        rH1, rH2, rH3 = Hessian(ry1, contribute, target), Hessian(ry2, contribute, target), Hessian(ry3, contribute, target)
        rE = cross(ry1, ry2, ry3, target)
        rLoss = np.hstack((rLoss, rE))
        counter = counter + 1
    
    #
    plt.figure(15)
    plt.plot(range(0, counter), rLoss)
    
    #
    # test data
    test = np.array(test)
    rat1, rat2, rat3 = activations(rw1, test, test), activations(rw2, test, test), activations(rw3, test, test)
    ryt1, ryt2, ryt3 = soft(rat1, rat2, rat3, test), soft(rat2, rat1, rat3, test), soft(rat3, rat2, rat1, test)
    ryt = np.hstack((ryt1, ryt2, ryt3))
    print(np.round(ryt))
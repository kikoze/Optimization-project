# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 19:38:32 2020

@author: Alexandre, Francisco, Miguel, Tiago
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# Load de dados
mat_file = 'data1.mat'

data = sio.loadmat(mat_file)
X = data['X']
Y = data['Y']

# Variáveis gerais
alfa_hat = 1
gama = 1e-4
beta = 0.5
n = X.shape[0]
K = X.shape[1]

s0 = np.array([-1, -1])
r0 = 0
epsilon = 1e-6

k = 0
s = s0
r = r0
grad = []

while True:
    df_ds = 0
    df_dr = 0

    for k in range(K):
        df_ds += (X[:,k]*np.exp(np.transpose(s) @ X[:,k]) - r)/(1 + 
            np.exp(np.transpose(s) @ X[:,k])) + Y[:,k]*X[:,k]
        df_dr += (-np.exp(np.transpose(s) @ X[:,k] - r))/(1 + 
            np.exp(np.transpose(s) @ X[:,k] - r)) + Y[:,k]
    df_ds /= K
    df_dr /= K
        
    g = np.sqrt(df_dr**2 + np.linalg.norm(df_ds)**2)
    grad.append(g)
    if g < epsilon:
        break

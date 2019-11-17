# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 03:48:05 2019

@author: akhil
"""

import numpy as np
from sklearn.datasets import load_boston as lb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def compute_cost(X,y,theta):
    m=len(y)
    h= X @ theta
    J =(1/(2*m))*np.sum((h-y)**2)
    
    return J

#Gradient descent baseed on no. of iterations
def gradient_descent(X,y,theta,alpha,n_iters):
    m=len(y)
    J_history =np.zeros((n_iters,1))
    for i in range(n_iters):
        theta = theta - (alpha/m)* X.T @ (X @ theta-y)
        J_history[i] = compute_cost(X,y,theta)
    return(J_history,theta)

#Gradient descent baseed on convergence limit
def gradient_descent_conv(X,y,theta,alpha,conv_limit):
    m=len(y)
    initial_cost = compute_cost(X,y,theta)
    J_history =np.array(initial_cost)
    delta=initial_cost
    while delta >= conv_limit:
        theta = theta - (alpha/m)* X.T @ (X @ theta-y)
        J_history = np.append(J_history,compute_cost(X,y,theta))
        delta = J_history[-2]-J_history[-1]
    return(J_history,theta)
   
def mean_norm(X):
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    X=(X-mu)/sigma
    return X

def plot_cost(J_history):
    plt.plot(range(len(J_history)), J_history, 'r')
    plt.title("Convergence Graph of Cost Function")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.show()
    
#loading dataset

dataset = lb()

X = dataset.data
y = dataset.target[:,np.newaxis]

X_sk = dataset.data
y_sk = dataset.target

# Number of samples in data
m = len(y)
#Normalizing the features
X= mean_norm(X)
# creating design matrix 
X = np.hstack((np.ones((m,1)),X))
n_f = np.size(X,1)
#Initializing theta's
theta = np.zeros((n_f,1))

alpha =.01
n_iters = 1500
conv_limit = 0.00001
# Solving Theta 
#(J_history,theta) = gradient_descent(X,y,theta,alpha,n_iters)
(J_history,theta) = gradient_descent_conv(X,y,theta,alpha,conv_limit)

# plotting cost function
plot_cost(J_history)

y_pred = X @ theta
score = 1 - (((y - y_pred)**2).sum() / ((y - y.mean())**2).sum())

# Used comparing our score with sklearn model
sklearn_regressor = LinearRegression().fit(X_sk, y_sk)
sklearn_accuracy = sklearn_regressor.score(X_sk, y_sk)
print(score,sklearn_accuracy)
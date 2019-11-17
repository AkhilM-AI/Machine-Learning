# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 00:52:20 2019

@author: akhil
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris as lr 
import seaborn as sns

#Sigmoid function 1/(1+exp(-z))
def sigmoid(z):
    return 1/(1+ np.exp(-z))

# predict class based on h values (h= sigmoid(X @ theta))
def predict(X, theta):
    return np.round(sigmoid(X @ theta))

#compute logistic cost function
#J(theta) = (1/m)*(-y.T * log(h) - (1-y).T * log(1-h) + (lamda/2)*theta_reg^2)
def compute_cost(X,y,theta,lamda):
    #since regularization factor is done for theta(0).
    theta_reg = np.vstack([0, theta[1:]])
    h=sigmoid(X@theta)
    J =-(1/m)*(np.dot(y.T,np.log(h)) + np.dot((1-y).T,np.log(1-h))-(lamda/2)*np.sum(theta_reg**2))
    return J

#Normalization of X
def mean_norm(X):
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    X=(X-mu)/sigma
    return X

#Gradient descent baseed on convergence limit
def gradient_descent_conv(X,y,theta,alpha,lamda,conv_limit):
    theta_reg = np.vstack([0, theta[1:]])
    m=len(y)
    initial_cost = compute_cost(X,y,theta,lamda)
    J_history =np.array(initial_cost)
    delta=initial_cost
    while delta >= conv_limit:
        # theta = theta - (alpha/m)*(X.T * (h-y) + lamda * theta_reg)
        theta = theta - (alpha/m)*(X.T @ (sigmoid(X@theta)-y) + lamda* theta_reg)
        J_history = np.append(J_history,compute_cost(X,y,theta,lamda))
        delta = J_history[-2]-J_history[-1]
    return(J_history,theta)


#Gradient descent baseed on no. of iterations
def gradient_descent(X,y,theta,alpha,lamda,n_iters):
    theta_reg = np.vstack([0, theta[1:]])
    m=len(y)
    J_history =np.zeros((n_iters,1))
    for i in range(n_iters):
        # theta = theta - (alpha/m)*(X.T * (h-y) + lamda * theta_reg)
        theta = theta - (alpha/m)*(X.T @ (sigmoid(X@theta)-y) + lamda* theta_reg)
        J_history[i] = compute_cost(X,y,theta,lamda)
    return(J_history,theta)

# Function to plot costfunction vs (# of iterations)    
def plot_cost(J_history):
    plt.plot(range(len(J_history)), J_history, 'r')
    plt.title("Convergence Graph of Cost Function")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.show()
    plt.clf()

# Main Function
if __name__ == "__main__": 
    #using load_iris dataset from sklearn
    dataset = lr()    
    #DataSet structure: { 'sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)'}
   # For ease of visualization considering 2 features {'sepal length (cm)','sepal width (cm)'}
    X = dataset.data[:, :2]
    #trying out 2 class model so converting target varible to 0 and 1's
    y = (dataset.target[:,np.newaxis] != 0) * 1    
    
    # Number of samples in data
    m = len(y)
    #Normalizing the features (scalling features)
    X_norm= mean_norm(X)
    # creating design matrix [1,X]
    X_dm = np.hstack((np.ones((m,1)),X_norm))
    n_f = np.size(X_dm,1)
    #Initializing theta's
    theta = np.zeros((n_f,1))
    #Model parameters
    lamda =1 ; alpha =.03 ; n_iters = 10000 ; conv_limit = 0.00001
    # Solving Theta 
    
#    (J_history,theta) = gradient_descent(X,y,theta,alpha,lamda,n_iters)
    (J_history,theta_optimal) = gradient_descent_conv(X_dm,y,theta,alpha,lamda,conv_limit)
    
    
    # plotting the cost function
    plot_cost(J_history)
    
    #Target values from the trained model
    y_pred = predict(X_dm, theta_optimal)
    
    # (fraction correct predictions)
    Accuracy = float(sum(y_pred == y))/ float(len(y))
    print(" Accuracy of the classification on training data: ",round(Accuracy*100,2))
    
    #plotting dataset along with the decision boundary from the model
    slope = -(theta_optimal[1] / theta_optimal[2])
    intercept = -(theta_optimal[0] / theta_optimal[2])
    
    plt.scatter(X_dm[:,1], X_dm[:,2],c=y.T[0], cmap='viridis')
    ax = plt.gca()
    ax.autoscale(False)
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + (slope * x_vals)
    plt.title("Dataset along with the decision boundary")
    plt.xlabel("Feature-1")
    plt.ylabel("Feature-2")
    plt.plot(x_vals, y_vals, c="k")
    

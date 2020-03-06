# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 09:28:21 2018

@author: ISHAN
"""

import numpy as np

def scaling(X):
    Xmax=np.max(X,axis=0)
    Xmin=np.min(X,axis=0)
    Xmean=np.mean(X,axis=0)
    X=np.divide(np.subtract(X,Xmean),np.subtract(Xmax,Xmin))
    return X

def hypo(theta,X):
    return np.dot(X,theta)

def costFunc(theta,X,Y,p):
    m1,n1=X.shape
    diff=np.subtract(hypo(theta,X),Y)
    diff=np.multiply(diff,diff)
    return np.sum(diff)/2/m1+np.sum(np.multiply(theta,theta))*p/2/m1

def derCost(theta,X,Y,p):
    m1,n1=X.shape
    diff=np.subtract(hypo(theta,X),Y)
    diff=np.dot(X.T,diff)
    diff=np.divide(diff,m1)
    diff=np.add(diff,np.multiply(theta,p/m1))
    diff[0]=diff[0]-theta[0]*p/m1
    return diff

def gradientDescent(theta,X,Y,alpha,p):
    deri=derCost(theta,X,Y,p)
    theta=np.subtract(theta,np.multiply(deri,alpha))
    return theta

trainingExample=np.loadtxt(fname='ex1data2.txt' ,delimiter=',')
m,n=trainingExample.shape
n=n-1
X=np.copy(trainingExample[:,:n])
Y=trainingExample[:,n]
for i in range(n):
    x=np.ones((m,1))
    for j in range(m):
        x[j]=X[j,i]
    X=np.concatenate((X,np.multiply(x,x)),axis=1)
X=scaling(X)
n=2*n
ones=np.ones((m,1))
X=np.concatenate((ones,X,),axis=1)
theta=np.zeros(n+1)
alpha=0.01
p=0.001    #lambda for Regularisation
iterations=1500
for i in range (iterations):
    theta=gradientDescent(theta,X,Y,alpha,p)
print(theta)
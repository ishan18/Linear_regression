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

def hypo(theta,X):
    return np.dot(X,theta)

def costFunc(theta,X,Y):
    m1,n1=X.shape
    diff=np.subtract(hypo(theta,X),Y)
    diff=np.multiply(diff,diff)
    return np.sum(diff)/2/m1

def derCost(theta,X,Y):
    m1,n1=X.shape
    diff=np.subtract(hypo(theta,X),Y)
    diff=np.dot(X.T,diff)
    return np.divide(diff,m1)

def gradientDescent(theta,X,Y,alpha):
    deri=derCost(theta,X,Y)
    theta=np.subtract(theta,np.multiply(deri,alpha))
    return theta

trainingExample=np.loadtxt(fname='ex1data2.txt' ,delimiter=',')
m,n=trainingExample.shape
n=n-1
X=np.copy(trainingExample[:,:n])
ones=np.ones((m,1))
scaling(X)
X=np.concatenate((ones,X,),axis=1)
Y=trainingExample[:,n]
theta=np.zeros(n+1)
alpha=0.01
iterations=1500
for i in range (iterations):
    theta=gradientDescent(theta,X,Y,alpha)
print(theta)
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 09:28:21 2018

@author: ISHAN
"""

import numpy as np

data=np.loadtxt('LinearRegressionData.txt' ,delimiter=',')
n=32
n2=2
m=47
def feat_scale(X,n,n2):
    f=np.random.rand(n,n2)
    for a in range(n2):
        f.T[a]=X.max(axis=0)[a]
    g=np.random.rand(n,n2)
    for a in range(n2):
        g.T[a]=X.min(axis=0)[a]
    h=np.random.rand(n,n2)
    for a in range(n2):
        h.T[a]=np.mean(X,axis=0)[a]
    X=np.divide((X-h),(f-g))
    return X;
    
def linear_reg_grad_descent(data,m,n,n2):
    X=data[:n,:n2]
    X=feat_scale(X,n,n2)
    X=np.concatenate((np.ones((n,1)),X),axis=1)
    Y=data[0:n,n2]
    theta=np.random.rand(n2+1,1)*10
    iteration=1500
    p=0.85
    j=0.001
    theta1=theta
    for i in range(iteration):
        for k in range(n2):
             s=0
             for l in range(n):
                 s=s+(np.dot(X[l],theta)-Y[l])*X[l,k]
             theta1[k]=theta[k]*(1-p*j/n)-p*s/n
        theta=theta1
    O=np.random.rand(m-n,1)
    X1=data[n:,:n2]
    X1=feat_scale(X1,m-n,n2)
    X1=np.concatenate((np.ones((m-n,1)),X1),axis=1)
    O=np.dot(X1,theta)
    Y1=data[n:,n2]
    #to calculate accuracy
    accuracy=np.random.rand(m-n,1)
    for b in range(m-n):
         accuracy[b]=((Y1[b]-O[b])*100/Y1[b])
         accuracy[b]=100-abs(accuracy[b])
    accuracy=np.sum(accuracy)/(m-n)
    return accuracy;
    
accuracy=linear_reg_grad_descent(data,m,n,n2)
print(accuracy)


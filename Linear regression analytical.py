# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 22:57:48 2018

@author: Ishan
"""
import numpy as np
import scipy.linalg as slin

data=np.loadtxt('LinearRegressionData.txt' ,delimiter=',')#dataset
n=32 #no. of training examples
n2=2 #no. of features
m=47 #total no. of datasets

#making a function for feature scaling
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
#making a function to carry out linear regression problem and 
#giving accuracy of the program by entering the required details
def linear_reg_analytical(data,m,n,n2):
    X=data[:n,:n2]
    #X=feat_scale(X,n,n2)
    X=np.concatenate((np.ones((n,1)),X),axis=1)#because X0=1
    Y=data[0:n,n2]
    XTX=np.dot(X.T,X)
    XTXinv=slin.inv(XTX)
    XTXinvXT=np.dot(XTXinv,X.T)
    theta=np.dot(XTXinvXT,Y)
    O=np.random.rand(m-n,1)
    X1=data[n:,:n2]
    #X1=feat_scale(X1,m-n,n2) 
    X1=np.concatenate((np.ones((m-n,1)),X1),axis=1)#because X0=1w
    O=np.dot(X1,theta)
    Y1=data[n:,n2]
    #to calculate accuracy
    accuracy=np.random.rand(m-n,1)
    for b in range(m-n):
         accuracy[b]=((Y1[b]-O[b])*100/Y1[b])
         accuracy[b]=100-abs(accuracy[b])
    accuracy=np.sum(accuracy)/(m-n)
    return accuracy;

accuracy=linear_reg_analytical(data,m,n,n2)
print('accuracy' ,accuracy)


#I have not used feature scaling here because i am getting more
#accuracy as compared to that with feature scaling.


    
    


    

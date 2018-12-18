# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 09:28:21 2018

@author: ISHAN
"""

import numpy as np

n=input('How many training examples you will provide?\n')
n2=input('How many features you will enter?\n')
n=int(n)
n2=int(n2)
X=np.random.rand(n,n2+1)
Y=np.random.rand(n,1)
print('"You are entering values for training set"')
for x in range(n):
    X[x:x+1]=1
    for a in range(n2):
        b=input('Enter feature: ')
        b=float(b)
        X[x:x+1,a+1]=b
    c=input('Enter result value: ')
    c=float(c)
    Y[x]=c
theta=np.random.rand(n2+1,1)
iteration=input('How many iterations you want to perform?\n')
iteration=int(iteration)
p=input('Enter alpha value: ')
p=float(p)
for i in range(iteration):
    for k in range(n2):
        s=0
        for l in range(n):
            s=s+(float(np.dot(X[l],theta))-Y[l])*X[l:l+1,k]      
        theta[k]=theta[k]-p/n*(s)
print(theta)
X1=np.random.rand(n2+1,1)
X1[0]=1
print('Your linear regression programme is ready')
k=input('How many try you will give to your problem? ')
k=int(k)
print('"Now you are entering values to calculate your result"')
for l in range(k):
    for a in range(n2):
         b=input('Enter Feature ')
         b=float(b)
         X1[a+1]=b
         output=np.dot(theta.T,X1)
    print('Your required value is=' ,float(output))

    
    



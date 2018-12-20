# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 09:28:21 2018

@author: ISHAN
"""

import numpy as np

data=np.array([2104,3,399900,
1600,3,329900,
2400,3,369000,
1416,2,232000,
3000,4,539900,
1985,4,299900,
1534,3,314900,
1427,3,198999,
1380,3,212000,
1494,3,242500,
1940,4,239999,
2000,3,347000,
1890,3,329999,
4478,5,699900,
1268,3,259900,
2300,4,449900,
1320,2,299900,
1236,3,199900,
2609,4,499998,
3031,4,599000,
1767,3,252900,
1888,2,255000,
1604,3,242900,
1962,4,259900,
3890,3,573900,
1100,3,249900,
1458,3,464500,
2526,3,469000,
2200,3,475000,
2637,3,299900,
1839,2,349900,
1000,1,169900,
2040,4,314900,
3137,3,579900,
1811,4,285900,
1437,3,249900,
1239,3,229900,
2132,4,345000,
4215,4,549000,
2162,4,287000,
1664,2,368500,
2238,3,329900,
2567,4,314000,
1200,3,299000,
852,2,179900,
1852,4,299900,
1203,3,239500,
])
#the data can be changed
n=input('How many training examples you will provide? ')
n=int(n) #total no. of training examples
n2=input('How many features you will provide? ') #total no. of features
n2=int(n2)
m=47 #total no. of data sets
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
    data=data.reshape(m,n2+1)
    data1=np.delete(data,n2,axis=1)
    X=data1[0:n]
    X=feat_scale(X,n,n2)
    X=np.concatenate((np.ones((n,1)),X),axis=1)
    Y=data[0:n,n2]
    theta=np.random.rand(3,1)*10
    iteration=input('How many iterations? ')
    iteration=int(iteration)
    p=input('Enter alpha value: ')
    p=float(p)
    j=input('Enter lambda value(for regularisation): ' )
    j=float(j)
    theta1=theta
    for i in range(iteration):
        for k in range(n2):
             s=0
             for l in range(n):
                 s=s+(float(np.dot(X[l],theta))-Y[l])*X[l:l+1,k]
             theta1[k]=theta[k]*(1-p*j/n)-(p/n)*s
             theta=theta1
    Y1=np.random.rand(m-n,1)
    X1=data1[n:m]
    X1=feat_scale(X1,m-n,n2)
    X1=np.concatenate((np.ones((m-n,1)),X1),axis=1)
    Y1=np.dot(X1,theta)
    Y=data[n:,n2]
    print(Y1)
    #to calculate accuracy
    accuracy=np.random.rand(m-n,1)
    for b in range(m-n):
         accuracy[b]=((Y[b]-Y1[b])/Y[b])*100
         accuracy[b]=100-abs(accuracy[b])
    accuracy=float(np.sum(accuracy))/(m-n)
    return accuracy;
    
accuracy=linear_reg_grad_descent(data,m,n,n2)
print(accuracy)


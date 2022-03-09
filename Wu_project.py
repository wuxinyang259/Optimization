# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize #minimize modeule
import numdifftools as nd
import random
os.chdir('C:\\Users\\18061\\OneDrive\\Desktop\\TTU course\\Optimization')

br=pd.read_csv('MiniProject1-Data.csv')

br.head()

br['BridgeCond']=br['BridgeCond'].replace(to_replace='S',value=1) #SAFE=1
br['BridgeCond']=br['BridgeCond'].replace(to_replace='U',value=0) #UNSAFE=0 

#choose variable
# approachwidth
# structure length
#Deck area 
# seismic hazard

# normalzation
Y=br['BridgeCond']

#br['Apprachwidth']=(br['Apprachwidth']-np.min(br['Apprachwidth']))/(np.max['Apprachwidth']-np.min(br['Apprachwidth']))
w, h = 1000,4
X = [[0 for x in range(w)] for y in range(h)] 

X[0][:]=br['ApproachWidth_M']
X[1][:]=br['StructureLength_M']
X[2][:]=br['DeckArea_SQM']
X[3][:]=br['SeismicHazard']

X[0][:]=(X[0][:]-np.min(X[0][:]))/(np.max(X[0][:])-np.min(X[0][:]))
X[1][:]=(X[1][:]-np.min(X[1][:]))/(np.max(X[1][:])-np.min(X[1][:]))
X[2][:]=(X[2][:]-np.min(X[2][:]))/(np.max(X[2][:])-np.min(X[2][:]))
X[3][:]=(X[3][:]-np.min(X[3][:]))/(np.max(X[3][:])-np.min(X[3][:]))

#Define a logistic regression function
X=np.array(X)



    
def LL(beta):

    y=beta[0]+beta[1]*X[0,:]+beta[2]*X[1,:]+beta[3]*X[2,:]+beta[4]*X[3,:]   
    #Lr_T=0 

    p_y=1/(1+np.exp(-y))

    Lr=-np.sum((Y*np.log(p_y))+((1-Y)*np.log(1-p_y)))

    return Lr


beta=[2,-1,-0.3,1,-0.5]
idx=0
tol=1e-5
eps=100000
while (eps>tol):
    a=nd.Gradient(LL)(beta)
    b=nd.Hessian(LL)(beta)
    binv=np.linalg.inv(b)
    eta=np.matmul(a,binv)
    beta_new=beta-eta
    eps=np.sum(np.sqrt((beta_new-beta)**2))
    eps=abs(eps)
    idx=idx+1
    beta=beta_new

beta_newton=beta


# Second method

beta=[2,-1,-0.3,1,-0.5]

obj= minimize(LL,beta,method='Nelder-Mead',options={'maxiter':10000})

print(obj.x)


#w1, h1 = 5000,5
#coe = [[0 for x in range(w1)] for y in range(h1)] 
#coe=np.array(coe)
#%%
# Resample function
def LL_samp(beta,X_samp,Y_samp):

    y_samp=beta[0]+beta[1]*X_samp[0,:]+beta[2]*X_samp[1,:]+beta[3]*X_samp[2,:]+beta[4]*X_samp[3,:]   
    #Lr_T=0 

    p_y_samp=1/(1+np.exp(-y_samp))

    Lr_samp=-np.sum((Y_samp*np.log(p_y_samp))+((1-Y_samp)*np.log(1-p_y_samp)))

    return Lr_samp

i=0
coe=[]

index=np.random.choice(range(0, 1000),1000)     
Y_samp=Y[index]
X_samp=X[:,index]
beta=[2,-1,-0.3,1,-0.5]
obj_sam= minimize(LL_samp,beta, args=(X_samp, Y_samp),method='Nelder-Mead',options={'maxiter':10000})

#coe = np.append(coe, obj_sam.x)
print(obj_sam.x)


#def nsquare(x, y):
  #  return (x*x + 2*x*y + y*y)
#print("The square of the sum of 2 and 3 is : ", nsquare(2, 3))



import numpy as np
import random
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker,cm
from matplotlib.ticker import LinearLocator,FormatStrFormatter
import matplotlib.mlab as mlab
from scipy.optimize import minimize, rosen, rosen_der,fmin_cg

def linearRegCostFunction(theta,X,y,lamda):
    J=0
    T=np.matrix(theta.reshape((np.shape(X)[1],1)))
    J=J+((1/(2*m))*(np.sum(np.power((np.matmul(X,T)-y),2))))
    J=J+((lamda/(2*m))*(np.sum(np.power(T[1:(np.shape(T)[0])],2))))
    return J

def linearRegGradFunction(theta,X,y,lamda):
    #theta=np.matrix(theta)
    #if np.shape(theta)[0]!=np.shape(X)[1]:
    #   theta=theta.T
    T=np.matrix(theta.reshape((np.shape(X)[1],1)))
    Grad=np.matrix(np.zeros(np.shape(T)))
    Grad=Grad+((1/m)*(np.matrix(np.sum(np.multiply(np.matmul(X,T)-y,X),axis=0)).T))
    Grad=Grad+((lamda/m)*(np.matrix(np.vstack((np.zeros((1,1)),T[1:np.shape(T)[0]])))))
    return np.squeeze(np.asarray(Grad))

def featureNormalize(X):
    mu=np.mat(np.mean(X,axis=0))
    sig=np.mat(np.std(X,axis=0))
    X_norm=X-mu/sig
    return [X_norm,mu,sig]
    
def trainLinearReg(X, y, lamda):
    ## Make sure that both costfunction and gradient function receives and returns flat array else convergence fails!!!!!!!!!!!!!!!!!!! 
    initial_theta=np.zeros(np.shape(X)[1])  
    #res = minimize(fun=linearRegCostFunction, x0=initial_theta, args=(X,y,lamda), method='CG', jac=linearRegGradFunction,options={'maxiter': 200, 'disp': True})
    res=fmin_cg(f=linearRegCostFunction,x0=initial_theta,fprime=linearRegGradFunction,args=(X,y,lamda),maxiter=200, disp= True)
    return np.matrix(res).T

file_name="ex5data1.mat"
data_content=sio.loadmat(file_name)
X=np.matrix(data_content['X'])
y=np.matrix(data_content['y'])
Xtest=np.matrix(data_content['Xtest'])
ytest=np.matrix(data_content['ytest'])
Xval=np.matrix(data_content['Xval'])
yval=np.matrix(data_content['yval'])
(m,n)=np.shape(X)


theta=np.array([[1.],[1.]])
temp_X=np.matrix(np.hstack((np.ones((m,1)),X)))
cost = linearRegCostFunction( theta, temp_X, y, 1)
print("Cost at theta = [1 ; 1]: (this value should be about 303.993192): ")
print(cost)

G = linearRegGradFunction( theta,temp_X, y, 1)
print("Gradient at theta = [1 ; 1]: (this value should be about [-15.303016; 598.250744]): ")
print(G)
#[fnx,mun,sigm]=featureNormalize(X)
#temp_X=np.matrix(np.hstack((np.ones((m,1)),fnx)))
lamda=0
t=trainLinearReg(temp_X,y,lamda)

print("Values of theta: ")
print(t)
predict=np.matmul(temp_X,t)



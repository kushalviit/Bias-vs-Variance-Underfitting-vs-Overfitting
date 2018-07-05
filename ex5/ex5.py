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
    X_norm=(X-mu)/sig
    return [X_norm,mu,sig]
    
def trainLinearReg(X, y, lamda):
    ## Make sure that both costfunction and gradient function receives and returns flat array else convergence fails!!!!!!!!!!!!!!!!!!! 
    initial_theta=np.zeros(np.shape(X)[1])  
    #res = minimize(fun=linearRegCostFunction, x0=initial_theta, args=(X,y,lamda), method='CG', jac=linearRegGradFunction,options={'maxiter': 200, 'disp': True})
    res=fmin_cg(f=linearRegCostFunction,x0=initial_theta,fprime=linearRegGradFunction,args=(X,y,lamda),maxiter=20000, disp= True)
    return np.matrix(res).T




def  polyFeatures(X, p):
     X_poly=np.matrix(np.zeros(np.shape(X)))+X
     for i in range(2,p+1):
         temp_x=np.power(X,i)
         X_poly=np.hstack((X_poly,temp_x))
     
     return X_poly


def learningCurve(X, y, Xval, yval, lamda):
    (m,n)=np.shape(X)
    error_train = np.matrix(np.zeros((m, 1)))
    error_val   = np.matrix(np.zeros((m, 1)))
    for i in range(m):
        t=trainLinearReg(X[0:i+1,:], y[0:i+1], lamda)
        J_train=linearRegCostFunction(t,X[0:i+1,:], y[0:i+1],0)
        J_val=linearRegCostFunction(t,Xval, yval,0)
        error_train[i]=J_train
        error_val[i]=J_val
    return [error_train,error_val]

def validationCurve(X, y, Xval, yval):
    lambda_vec = np.array([0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10])
    error_train = np.zeros((len(lambda_vec), 1))
    error_val = np.zeros((len(lambda_vec), 1))
    i=0
    for lamda in lambda_vec:
        t=trainLinearReg(X,y,lamda)
        J_train=linearRegCostFunction(t,X,y,lamda)
        J_val=linearRegCostFunction(t,Xval,yval,lamda)
        error_train[i]=J_train
        error_val[i]=J_val
        i=i+1
    return [lambda_vec, error_train, error_val]

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
temp_Xval=np.matrix(np.hstack((np.ones((np.shape(Xval)[0],1)),Xval)))
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

lamda=0
[error_train, error_val] =learningCurve(temp_X, y, temp_Xval, yval,lamda)

train_samples=np.arange(1,m+1)
plt.plot(train_samples,error_train)
plt.plot(train_samples,error_val)
plt.show()

p=8
X_p=polyFeatures(X,p)

[norm_X_p,me,si]=featureNormalize(X_p)

norm_X_p=np.matrix(np.hstack((np.ones((np.shape(X)[0],1)),norm_X_p)))

X_p_test=polyFeatures(Xtest,p)
norm_X_p_test=(X_p_test-me)/si
norm_X_p_test=np.matrix(np.hstack((np.ones((np.shape(Xtest)[0],1)),norm_X_p_test)))


X_p_val=polyFeatures(Xval,p)
norm_X_p_val=(X_p_val-me)/si
norm_X_p_val=np.matrix(np.hstack((np.ones((np.shape(Xval)[0],1)),norm_X_p_val)))



lamda=0
t=trainLinearReg(norm_X_p,y,lamda)
#print(t)
predict=np.matmul(norm_X_p,t)

[error_train_p, error_val_p] =learningCurve(norm_X_p, y, norm_X_p_val, yval,lamda)

train_samples=np.arange(1,m+1)
plt.plot(train_samples,error_train_p)
plt.plot(train_samples,error_val_p)
plt.show()


[lambda_vec, error_train_p, error_val_p] = validationCurve(norm_X_p, y, norm_X_p_val, yval)



plt.plot(lambda_vec,error_train_p)
plt.plot(lambda_vec,error_val_p)
plt.show()

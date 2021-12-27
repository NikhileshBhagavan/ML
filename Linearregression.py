import sklearn as sk
import matplotlib.pyplot as plt
import numpy as np
import random as r
from sklearn import model_selection
from sklearn import linear_model

X=list(range(0,10))
Y=[]
for k in X:
    Y.append(1.8*k+32+r.random())

plt.plot(X,Y,"-*r")#X and Y here must be iterable 1dimensional
plt.show()
X=np.array(X).reshape(-1,1)#X must be 2d numpy array
Y=np.array(Y).reshape(-1,1)#Y can be 1d or 2d numpy array
xtrain,xtest,ytrain,ytest=model_selection.train_test_split(X,Y,test_size=0.3)//returns all are nparrays
xtest=np.array([[0],[100]])
ytest=np.array([[32],[212]])
print(xtrain)
print(xtest)
print(ytrain)
print(ytest)
model=linear_model.LinearRegression()
model.fit(xtrain,ytrain)
accuracy=model.score(xtest,ytest)
print(accuracy*100)
print(model.coef_,model.intercept_)
x=model.predict(np.array([[1000]]))#input must be 2d numpy array as X and output will be 1d numpy array
print(x)


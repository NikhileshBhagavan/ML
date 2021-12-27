import pandas as pd
import numpy as np
from sklearn import *
data=pd.read_csv("Hyderabad_rent.csv")
print(data.info())
data.drop(["seller_type","layout_type","property_type"],axis="columns",inplace=True)
print(data.info())
data.dropna(inplace=True)
print(data.info())
print(data.head())
l=[]
for k in data["locality"]:
    l.append(k)
res = []
for i in l:
    if i not in res:
        res.append(i)

print(res)
d=dict(zip(res,range(1,len(res)+1)))
print(d)
def convertlocalitytonumber(s):
    return d[s]
data["locality"]=data["locality"].map(convertlocalitytonumber)
print(data["locality"])
def removecommas(s):
    for k in s:
        if k==",":
            return s[0:s.index(k)]+s[s.index(k)+1:]

    return s

data["price"]=data["price"].map(removecommas)
def furnish(s):
    if s=="Unfurnished":
        return 0
    if s == "Semi-Furnished":
        return 1
    if s == "Furnished":
        return 2

data["furnish_type"]=data["furnish_type"].map(furnish)
print(data["furnish_type"])
def bathroomconverter(s):
     return int(s[0:1])
for x in data.index:
    try:
        int(data.loc[x, "bathroom"][0:1])
    except:
        data.drop(x, inplace = True)
data["bathroom"]=data["bathroom"].map(bathroomconverter)
print(data["bathroom"])

x=np.array(data.drop(["price"],axis="columns"))
y=np.array(data["price"])
print(x,y)
print(data.info())
xtrain,xtest,ytrain,ytest=model_selection.train_test_split(x,y,test_size=0.002)
print(len(xtrain))#xtrain is np.array 2d
model=linear_model.LinearRegression()
model.fit(xtrain,ytrain)
accuracy=model.score(xtest,ytest)
print(accuracy*100)
print(model.coef_)
print(model.predict(np.array([[1,1,320,2,1],[2,3,200,2,1]])))


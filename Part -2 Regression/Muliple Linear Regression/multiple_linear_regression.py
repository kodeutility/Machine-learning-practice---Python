#Mulitple Linear Regression

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import data
dataset = pd.read_csv("50_Startups.csv")

#creating matrix of features
X = dataset.iloc[:,:-1].values
#creating matrix of dependent variables
y = dataset.iloc[:,4].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()

#Encoding Independent Variable
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding dummy Variable trap
X = X[:,1:]

#split dataset to training set and test set
#set 20% as test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Fiiting  multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the test set results
y_pred = regressor.predict(X_test)
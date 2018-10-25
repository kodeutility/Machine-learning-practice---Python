#Simple linear regression

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import data
dataset = pd.read_csv("Salary_Data.csv")

#creating matrix of features
X = dataset.iloc[:,:-1].values
#creating matrix of dependent variables
y = dataset.iloc[:,1].values

#split dataset for training set and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1/3,random_state=0)

#Fitting Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the test set results
y_pred = regressor.predict(X_test)

#Visualising the Training set results
plt.scatter(X_train,y_train,color = "red")
plt.plot(X_train,regressor.predict(X_train),color = "blue")
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#Visualising the Test set results
plt.scatter(X_test,y_test,color = "red")
plt.plot(X_train,regressor.predict(X_train),color = "blue")
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#polynomial linear regression

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import data
dataset = pd.read_csv("Position_Salaries.csv")

#creating matrix of features
X = dataset.iloc[:,1:2].values
#creating matrix of dependent variables
y = dataset.iloc[:,2].values

#Fitting polynomial regression to the dataset
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly,y)

#Visualising the polynomial regression results
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid)),1)
plt.scatter(X,y,color = "red")
plt.plot(X_grid,lin_reg.predict(poly_reg.fit_transform(X_grid)),color = "blue")
plt.title("Position Salaries graph")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

#Predicting a new value
lin_reg.predict(poly_reg.fit_transform(6.5))
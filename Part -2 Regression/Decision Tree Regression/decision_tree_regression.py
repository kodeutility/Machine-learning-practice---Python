#Decision tree regression

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

#Fitting Random Forest Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

#Visualising the Random Forest Regression results
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid)),1)
plt.scatter(X,y,color = "red")
plt.plot(X_grid,regressor.predict(X_grid),color = "blue")
plt.title("Position Salaries graph")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

#Predicting a new Salary
y_pred = regressor.predict(6.5)


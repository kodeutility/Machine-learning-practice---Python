#Data preprocessing 

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import data
dataset = pd.read_csv("Data.csv")

#creating matrix of features
X = dataset.iloc[:,:-1].values
#creating matrix of dependent variables
y = dataset.iloc[:,3].values

#handling missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN",strategy="mean",axis=0)

#fit imputer object to our matrix
imputer = imputer.fit(X[:,1:3])

#replace missing data with mean
X[:,1:3] = imputer.transform(X[:,1:3])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
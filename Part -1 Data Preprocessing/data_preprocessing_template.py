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
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#split dataset to training set and test set
#set 20% as test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test =sc_X.transform(X_test)
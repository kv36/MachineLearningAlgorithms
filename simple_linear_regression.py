#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 15:18:59 2017

@author: kartheekvadlamani
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')
X_sln = dataset.iloc[:, :-1].values
Y_sln = dataset.iloc[:, 1].values

# Splitting the Data Set into Training Set and Testing Set
from sklearn.cross_validation import train_test_split
X_sln_train, X_sln_test, Y_sln_train, Y_sln_test = train_test_split(X_sln, Y_sln, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting a simple linear regression Model to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_sln_train, Y_sln_train)


# Predicting the Test set results
y_sln_pred = regressor.predict(X_sln_test)

# Visualising the Training set results
plt.scatter(X_sln_train, Y_sln_train, color = 'red')
plt.plot(X_sln_train, regressor.predict(X_sln_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# Visualising the Test set results
plt.scatter(X_sln_test, Y_sln_test, color = 'red')
plt.plot(X_sln_train, regressor.predict(X_sln_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
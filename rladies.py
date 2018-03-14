#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:38:01 2018

@author: sedacavdaroglu
"""

#Let's start the clock to time the Python code.

import time
start_time = time.time()

#Read the data.
import pandas as pd
train_data = pd.read_csv("/Users/sedacavdaroglu/Desktop/DATS_Final-master/train_housing.csv")
test_data = pd.read_csv("/Users/sedacavdaroglu/Desktop/DATS_Final-master/test_housing.csv")

#Let's check the first 10 rows of the data to get an idea.
train_data.head(n=10)

#Let's visualize the relationship between selling price and other numerical variables with a heatmap.
num_data = train_data[['saleprice','lot.frontage','lot.area','full.bath','bedroom.abvgr','yr.sold']]
import seaborn as sns


# calculate the correlation matrix
corr = num_data.corr()

# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)

#Now, let's do some regression analysis.
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

y=num_data['saleprice']
X=num_data.drop('saleprice',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Scale the data
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

#do the prediction with Lasso with Least Angle Regression
lars_reg=LassoCV(alphas = np.linspace(0.001, 1000, num=10000)).fit(X_test, y_test)
lars_reg.fit(X_train,y_train)
print(lars_reg.coef_)

coef = pd.Series(lars_reg.coef_, index = X.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

#Let's check the coefficients to see which parameters lasso picked.
import matplotlib 
import matplotlib.pyplot as plt
imp_coef = pd.concat([coef.sort_values().head(5),
                     coef.sort_values().tail(0)])
    
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the LassoLars Model")

#Let's try the machine learning package scikitlearn to do some simple knn. Let's pick a more intuitive dataset for knn.
from sklearn import datasets
iris = datasets.load_iris()
#convert into data frame
df =  pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

#Split the data into training and test sets.
y=df['target']
X=df.drop('target',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Let's do the kn with 10 cross validations.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# instantiate learning model with auto option (picks the best algorithm for knn)
knn = KNeighborsClassifier(algorithm='auto')

# fit the model
knn.fit(X_train, y_train)

# predict the response
pred = knn.predict(X_test)

#Let's check the accuracies.
# evaluate accuracy
print(accuracy_score(y_test, pred))

#How long does this all take?
end_time = time.time()
print(end_time-start_time)
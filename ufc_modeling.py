#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 23:25:07 2020

@author: aidanosullivan
"""
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.max_seq_items = 200
from matplotlib import pyplot as plt

from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

ufc_official = pd.read_csv("/Users/aidanosullivan/Desktop/UCLA_Extension/Mini_Project/MiniProject_Final/ufc_offical.csv")

y = ufc_official.Winner #label
x = ufc_official.drop('Winner', axis = 1) #features

#drop categorical variables that will be impossible to get dummies for
x = x.drop(['R_fighter', "B_fighter", "location", "country", "date", "weight_class"], axis = 1)

#get dummy variables
x = pd.get_dummies(x, columns = ["title_bout", "gender", "better_rank", "B_Stance", "R_Stance"], drop_first=True)
y = pd.get_dummies(y, columns = 'Winner', drop_first = True)


#split data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

#Scale the variables

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Perform Feature Selection - this could take a while!!
rfecv = RFECV(estimator = RandomForestClassifier(), cv = 5, step =1)
rfecv = rfecv.fit(x_train, y_train.values.ravel())
f  = rfecv.get_support(1)
f

x_train = pd.DataFrame(x_train)
x_train = x_train[x_train.columns[f]] #subset only the best columns of x_train
#x_train now has 20 columns...honestly that seems like a lot


rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
rf.fit(x_train, y_train.values.ravel())

y_train_pred = rf.predict(x_train)
print('training set accuracy %2.2f' % (accuracy_score(y_train, y_train_pred)*100))
print("training set f1 %2.2f" % (f1_score(y_train, y_train_pred)*100))

#now do the same thing but to find testing accuracy

x_test = pd.DataFrame(x_test)
x_test = x_test[x_test.columns[f]]

y_test_predict = rf.predict(x_test)

print('training set accuracy %2.2f' % (accuracy_score(y_test, y_test_predict)*100))
print("training set f1 %2.2f" % (f1_score(y_test, y_test_predict)*100))
#yikes so after 100% training accuracy, we got 63.90 testing accuracy

#Let's try rewritting some of the above code to reduce overfitting. If it doesn't work the first 
#time, then I will just edit the code from here

x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, test_size = 0.3)
scaler = StandardScaler()
x_train1 = scaler.fit_transform(x_train1)
x_test1 = scaler.transform(x_test1)
#we'll also rerun feature selection, in hope that we will get fewer predictors
rfecv = RFECV(estimator = RandomForestClassifier(), cv = 10, step =1)
rfecv = rfecv.fit(x_train1, y_train1.values.ravel())
g  = rfecv.get_support(1)

#fuck we now have wayyy more predictors - all 69 to be exact
#let's NOT use g
#subset x_train1 by the good predictors
x_train1 = pd.DataFrame(x_train1)
x_train1 = x_train1[x_train1.columns[f]]
#create another object of class random forest classifier, this time with more trees
#also i'll change the max_depth, and the max_features (which is like mtry in R)
rf = RandomForestClassifier(n_estimators = 2000, random_state = 42, max_depth = 10, max_features=int(math.sqrt(len(f))))
rf.fit(x_train, y_train.values.ravel())
#predict the y_train1 labels to get the training error
y_train_pred1 = rf.predict(x_train1)
print('training set accuracy %2.2f' % (accuracy_score(y_train1, y_train_pred1)*100))
print("training set f1 %2.2f" % (f1_score(y_train, y_train_pred)*100))
#Do the same thing but for testing error
x_test1 = pd.DataFrame(x_test1)
x_test1 = x_test1[x_test1.columns[f]]
#predict the y_test1 labels to get the testing error
y_test_predict1 = rf.predict(x_test1)
print('testing set accuracy %2.2f' % (accuracy_score(y_test1, y_test_predict1)*100))
print("testing set f1 %2.2f" % (f1_score(y_test1, y_test_predict1)*100))
#final testing error is 83.23% - not bad for a little tweeking. more will come


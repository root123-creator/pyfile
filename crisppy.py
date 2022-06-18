#!/usr/bin/env python
# coding: utf-8

import pandas as pd


# # Data importing

cdata = pd.read_csv('./data.csv')



cdata


# # Data Visualisation



import matplotlib.pyplot as plt



np.random.seed(10)
plot1 = cdata['Floor Space']
 
fig = plt.figure(figsize =(8, 4))
 
plt.boxplot(plot1)
 
plt.show()



plot2=cdata.plot.hist(bins=15,alpha=0.7)
plot2;



import seaborn as sns



corrmat = cdata.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
g=sns.heatmap(cdata[top_corr_features].corr(),annot=True,cmap="magma")


# # Data preparation


cdata.isna().sum()


cdata = cdata.apply(lambda x: pd.factorize(x)[0])


cdata.head()


y= cdata['Performance']


X = cdata.drop(['Performance'], axis = 1)


# # Model implementation

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import tree



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 100)


dec = DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=3, min_samples_leaf=5)


dec.fit(X_train, y_train)


y_pred = dec.predict(X_test)


# # Accuracy scores

accuracy_score(y_test, y_pred)


print(classification_report(y_test, y_pred))


tree.plot_tree(dec);

from sklearn.ensemble import RandomForestClassifier

random = RandomForestClassifier(n_estimators = 100) 


random.fit(X_train, y_train)


y_pred = random.predict(X_test)


#Evaluation 2


accuracy_score(y_test, y_pred)


print(classification_report(y_test, y_pred))


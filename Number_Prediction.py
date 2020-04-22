# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:59:11 2020

@author: Harshil
"""

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#Importing Dataset
train_file=pd.read_csv('train.csv')
test_file=pd.read_csv('test.csv')

np.sort(train_file.label.unique())

num_train,num_validation=int(len(train_file)*0.8),int(len(test_file)*0.2)
print(num_train,num_validation)

X_train=train_file.iloc[:num_train,1:].values
X_label=train_file.iloc[:num_train,0].values
X_validation=train_file.iloc[:num_validation,1:].values
y_validation=train_file.iloc[:num_validation,0].values
#print(X_train.shape)
print(X_validation.shape)

#Visualizing training data
index=3
print('Label' + str(X_label[index]))
plt.imshow(X_train[index].reshape((28,28)),cmap="gray")
plt.show()

#Applying Random Forest Classifier
clf=RandomForestClassifier()
clf.fit(X_train,X_label)

#Predicting results and analying the results
prediction_validation=clf.predict(X_validation)
print("Validation Accuracy" + str(accuracy_score(y_validation,prediction_validation)))

X_test=test_file
prediction_test=clf.predict(X_test)
print("Predicted" + str(prediction_test[index]))
plt.imshow(X_test.iloc[index].values.reshape((28,28)),cmap="gray")

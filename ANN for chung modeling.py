# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 10:31:54 2020

@author: admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import data set 
dataset = pd.read_csv("D:\ML\dataset\Churn_Modelling.csv")
X = dataset.iloc[:,3:13]
y = dataset.iloc[:,13]

# Create dummy variables
geography = pd.get_dummies(X['Geography'],drop_first=True)
gender = pd.get_dummies(X['Gender'],drop_first=True)

# concatinate with data frame

X = pd.concat([X,geography,gender],axis=1)

# drop unnecessary columns
X = X.drop(['Geography','Gender'],axis=1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# ANN Library 
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

# Intialising the ANN
classifier = Sequential()

# adding the input layer and first hidden layer
classifier.add(Dense(units = 10,kernel_initializer = 'he_normal',activation='relu',input_dim=11))
classifier.add(Dropout(0.3))
# adding the another input layer
classifier.add(Dense(units = 20,kernel_initializer = 'he_normal',activation='relu'))
classifier.add(Dropout(0.4))

# adding the another input layer
classifier.add(Dense(units = 15,kernel_initializer = 'he_normal',activation='relu'))
classifier.add(Dropout(0.2))

# adding the another input layer
classifier.add(Dense(units = 1,kernel_initializer = 'glorot_uniform',activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])

# fitting ANN to the training dataset
model_history=classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 10,epochs = 100)


# Predicting the test result
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making Confusion Matrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)

# Calculating Accuracy score
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred,y_test)

print(model_history.history.keys())
# summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

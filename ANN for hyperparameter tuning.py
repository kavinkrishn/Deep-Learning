# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 18:37:38 2020

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

## perform hyperparameter Optimization

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Embedding,LeakyReLU,BatchNormalization,Flatten
from keras.activations import relu,sigmoid

def create_model (layers,activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i == 0:
            model.add(Dense(nodes,input_dim=X_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
    model.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))
    
    model.compile(optimizer='adma',loss='binary_crossentropy',metrics=['accuracy'])
    return model
model = KerasClassifier(build_fn=create_model,verbose=0)

layers = [[20],[40,20],[45,30,15]]
activations =['sigmoid','relu']
param_grid = dict(layers=layers,activation=activations,batch_size=[128,256],epochs=[])
grid = GridSearchCV(estimator=model,param_grid=param_grid,cv=5)

grid_result = grid.fit(X_train,y_train)

print(grid_result.best_score_,grid_result.best_params_)

pred_y = grid.predict(X_test)
y_pred = (pred_y > 0.5)

from sklearn.metrics import accuracy_score,confusion_matrix
cm = confusion_matrix(y_pred,y_test)
score = accuracy_score(y_pred,y_tests)
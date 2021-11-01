# -*- coding: utf-8 -*-
"""
Created on Mon May 11 20:05:33 2020

@author: Pavan
"""
import pandas as pd 
from sklearn.model_selection import train_test_split

header_list = ["A", "B", "C","D","E", "F", "G","H","I", "J", "K","L","M", "N", "O","P","Q", "R", "S","T","U", "V", "W","X","Y","Z","A1", "B1", "C1","D1","E1", "F1", "G1","H1","I1", "J1", "K1","L1","M1", "N1", "O1","P1","Q1", "R1", "S1","T1","U1", "V1", "W1","X1","Y1","Z1","A2", "B2", "C2","D2","E2", "F2", "G2","H2","I2", "J2", "K2","L2","M2", "N2", "O2","P2","Q2", "R2", "S2","T2","U2", "V2", "W2","X2","Y2","Z2","Result"]


df = pd.read_csv("dataset.csv", names=header_list)

X_train_1=df.iloc[:,:-1]
Y_train_1=df.iloc[:,-1]
#train_dataset = pd.read_csv('dataset.csv', delimiter = ',', quoting = 3)
X_train, X_test, y_train, y_test = train_test_split(X_train_1,Y_train_1, test_size=0.2)

from sklearn import svm
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)

#predicting Y values for test data
Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, Y_pred)



# -*- coding: utf-8 -*-
"""
Created on Mon May 11 20:05:33 2020

@author: Pavan
"""


import pickle

header_list =['Mean_R', 'Mean_G','Mean_B','Standard_deviation_R', 'Standard_deviation_G','Standard_deviation_B','Skewness_R', 'Skewness_G','Skewness_B','Kurtosis_R', 'Kurtosis_G','Kurtosis_B','Entropy_R', 'Entropy_G','Entropy_B','Range_R', 'Range_G','Range_B','ASM_0', 'Contrast_0','Corrleation_0','entropy_0','ASM_45', 'Contrast_45','Corrleation_45','entropy_45','ASM_90', 'Contrast_90','Corrleation_90','entropy_90','ASM_135', 'Contrast_135','Corrleation_135','entropy_135','Short_run_emphasis_0', 'Long_run_emphasis_0','Gray_level_nonuniformity_0' ,'Run_length_nonuniformity_0','Run_percentage_0','Low_gray_level_run_emphasis_0','High_gray_level_run_emphasis_0','Short_run_Low_gray_level_emphasis_0','Short_run_High_gray_level_emphasis_0','Long_run_Low_gray_level_emphasis_0','Long_run_High_gray_level_emphasis_0','Short_run_emphasis_45', 'Long_run_emphasis_45','Gray_level_nonuniformity_45' ,'Run_length_nonuniformity_45','Run_percentage_45','Low_gray_level_run_emphasis_45','High_gray_level_run_emphasis_45','Short_run_Low_gray_level_emphasis_45','Short_run_High_gray_level_emphasis_45','Long_run_Low_gray_level_emphasis_45','Long_run_High_gray_level_emphasis_45','Short_run_emphasis_90', 'Long_run_emphasis_90','Gray_level_nonuniformity_90' ,'Run_length_nonuniformity_90','Run_percentage_90','Low_gray_level_run_emphasis_90','High_gray_level_run_emphasis_90','Short_run_Low_gray_level_emphasis_90','Short_run_High_gray_level_emphasis_90','Long_run_Low_gray_level_emphasis_90','Long_run_High_gray_level_emphasis_90','Short_run_emphasis_135', 'Long_run_emphasis_135','Gray_level_nonuniformity_135' ,'Run_length_nonuniformity_135','Run_percentage_135','Low_gray_level_run_emphasis_135','High_gray_level_run_emphasis_135','Short_run_Low_gray_level_emphasis_135','Short_run_High_gray_level_emphasis_135','Long_run_Low_gray_level_emphasis_135','Long_run_High_gray_level_emphasis_135','Result']

import pandas as pd 
from sklearn.model_selection import train_test_split
df = pd.read_csv("dataset.csv", names=header_list)

X_train_1=df.iloc[:,:-1]
Y_train_1=df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X_train_1,Y_train_1, test_size=0.2)

from sklearn import svm
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)

#predicting Y values for test data
Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, Y_pred)

pickle.dump(classifier, open(r'C:\Users\Pavan\Desktop\Project\corrosion_detection\predictor\model\model.pkl', 'wb'))

'''
X_train_1=df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]]
Y_train_1=df.iloc[:,-1]
from sklearn import svm
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)

'''




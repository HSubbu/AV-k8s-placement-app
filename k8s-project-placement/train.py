#!/usr/bin/env python
# coding: utf-8

# ## ML_Zoomcamp October 2021  PROJECT 1 by Subramanian Hariharan
# Problem Statement
# A campus recruitment program is a program conducted within universities or other educational institutions that helps students as they near graduation find employment. As part of this program, educational institutions partner with corporations that wish to recruit from student populations.
# The placement of a student depends on various factors like their performance in school and colleges, work experience, etc.  
#  The objective here is to predict if a student will get placed or not.
#import all libraries
import pandas as pd 
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

#import train and test dataset
print('Reading Training CSV files...')
df_full_train = pd.read_csv('Train_Data.csv')
df_test = pd.read_csv('Test_Data.csv')
#converting target variable to numeric
print('Converting target variable into numeric ..')
df_full_train.status = (df_full_train.status == 'Placed').astype(int)

# we will use dictvectorizer to convert cat columns to numeric
dv = DictVectorizer(sparse=False)
#Fit the tuned RF on entire training dataset and use it for prediction on test datatset
print('Fitting the Random Forest Model with cross validation...')
model_rf = RandomForestClassifier(max_depth=16, n_estimators= 512)

kfold = KFold(n_splits=5)
kf_score =[]
for train_idx,val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
    X_train = df_train.drop('status',axis=1)
    y_train = df_train.status.values
    X_val = df_val.drop('status',axis=1)
    y_val = df_val.status.values
    train_dict = X_train.to_dict(orient='records')
    X_train_encoded = dv.fit_transform(train_dict)
    val_dict = X_val.to_dict(orient='records') #applying ohe for val data
    X_val_encoded = dv.transform(val_dict)
    model_rf.fit(X_train_encoded,y_train)
    y_pred = model_rf.predict_proba(X_val_encoded)[:, 1] #prediction of val set
    loop_score = accuracy_score(y_val, y_pred >= 0.5)
    kf_score.append(loop_score)

print(f'KFOLD mean score {np.mean(kf_score)}')   

# create a full training dataset 
X_full_trg = df_full_train.drop('status',axis=1)
y_full_trg = df_full_train['status']
full_trg_dicts =  X_full_trg.to_dict(orient='records')
X_full_trg_encoded = dv.fit_transform(full_trg_dicts)
#fitting on whole trg set
print('Fitting model on whole training set ..')
model_rf.fit(X_full_trg_encoded,y_full_trg)

#saving the model and dv as pickle
print('Saving pickle file...')
with open("project_one_model.pkl","wb") as f_out:
    pickle.dump((dv,model_rf),f_out)

print('Model saved to project_one_model.pkl')
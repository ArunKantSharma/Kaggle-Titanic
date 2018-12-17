#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 10:30:24 2018

@author: arunkantsharma


Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination

"""
#Main modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Data Processing 

#Import Data
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

#Seperate Dependent and Independent variable from Training set,
# in given example test set have no Dependent Variable
X = train.loc[:, train.columns != 'Survived']
Y = train.loc[:, train.columns == 'Survived']

#Handle missing data and drop unnecessary columns

#Categorise name by Title 
X['Title'] = X['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2,"Master": 3, "Dr": 4, 
                 "Rev": 4, "Col": 4, "Major": 4, "Mlle": 4,
                 "Countess": 4,"Ms": 4, "Lady": 4, "Jonkheer": 4, 
                 "Don": 4, "Dona" : 4, "Mme": 4,"Capt": 4,"Sir": 4 }

X['Title'] = X['Title'].map(title_mapping)
test['Title'] = test['Title'].map(title_mapping)

#drop unnecessary columns = 'Name'
X.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

#categorize Data
#categorize Sex
sex_mapping = {'male':0, 'female': 1}
X['Sex'] = X['Sex'].map(sex_mapping) 
test['Sex'] = test['Sex'].map(sex_mapping)

#Missing Data 
#Replace missing age by median of missing age by 'Title'
X["Age"].fillna(X.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)


#Missing Embarked 
# S    644, C    168, Q     77
#No Missing Embarked in test dataset

X['Embarked'] = X['Embarked'].fillna('S')

#categorize Embarked
embarked_mapping = {'S':0, 'C':1, 'Q':2 }
X['Embarked'] = X['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)


#X['Cabin'].astype(str).str[0].value_counts()
#X_Temp =X[pd.notna(X['Cabin']) ==True]
#X_Temp['Cabin_code'] = X_Temp.loc[:,'Cabin'].astype(str).str[0]
#X_Temp.groupby(['Pclass','Cabin_code']).size()
#1 -->C , 2--> F, 3--> Empty F

#Droping Cabin column, 
#we can impute missing Cabin values by Family member , Fare and PClass
#But now I am dropping it, I will use it later if I don't get good result.

#drop unnecessary columns = 'Cabin'
X.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)

#drop unnecessary columns = 'Ticket'
X.drop('Ticket', axis=1, inplace=True)
test.drop('Ticket', axis=1, inplace=True)

#drop unnecessary columns = 'PassengerId'
X.drop('PassengerId', axis=1, inplace=True)
test.drop('PassengerId', axis=1, inplace=True)

#Replace missing age by median of missing age by 'Title'
X["Fare"].fillna(X.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)



#Categorize Age
X_test_data = [X,test]
for ds in X_test_data:
    ds.loc[ ds['Age'] <= 16, 'Age'] = 0,
    ds.loc[(ds['Age'] > 16) & (ds['Age'] <= 26), 'Age'] = 1,
    ds.loc[(ds['Age'] > 26) & (ds['Age'] <= 36), 'Age'] = 2,
    ds.loc[(ds['Age'] > 36) & (ds['Age'] <= 62), 'Age'] = 3,
    ds.loc[ ds['Age'] > 62, 'Age'] = 4

#Categorize Age
for ds in X_test_data:
    ds.loc[ ds['Fare'] <= 17, 'Fare'] = 0,
    ds.loc[(ds['Fare'] > 17) & (ds['Fare'] <= 30), 'Fare'] = 1,
    ds.loc[(ds['Fare'] > 30) & (ds['Fare'] <= 100), 'Fare'] = 2,
    ds.loc[ ds['Fare'] > 100, 'Fare'] = 3

#Create FamilySize instead of SibSp and Parch    
X['FamilySize'] = X.SibSp + X.Parch + 1
test['FamilySize'] = test.SibSp + test.Parch + 1

X.drop('SibSp', axis=1, inplace=True)
test.drop('SibSp', axis=1, inplace=True)

X.drop('Parch', axis=1, inplace=True)
test.drop('Parch', axis=1, inplace=True)

#Splitting Dataset X into X_training set and X_test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = .2)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear')
classifier.fit(X_train, y_train)


#Predecting the test set result
y_pred = classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''
cm = 103  14
    16 46
    that is approx 83% accuracy. Better than other previous model/assumptions.
'''

# Creat prediction for 'test' Dataset
test_predict = pd.DataFrame(classifier.predict(test))
test_predict.rename(columns={0: 'Survived'}, inplace=True)

test_new = pd.read_csv('input/test.csv')
test_new = test_new.PassengerId
result = pd.concat([test_new, test_predict], axis=1)

result.to_csv('submission.csv', index=False)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 00:51:17 2018

@author: manoj
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 05:58:51 2018

@author: manoj
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ds_train = pd.read_csv('train.csv')
ds_test = pd.read_csv('test.csv')



ds_test_copy = ds_test.copy(deep=True)


#get train data info so that we can find missing data and datatypes
#print(ds_train.info())
#print('-'*10)
#print(ds_test.info())



temp = ds_train.describe(include='all')

#print(ds_train.sample(10))


#find the total number of null value rows for each column
#print(ds_train.isnull().sum())

#print(ds_test.isnull().sum())

#combine both the data sets so that we can fill empty values
dataset = [ds_train, ds_test]

#Need to preprocess the data, Here we can see that there are null values in Age, Cabin and Embarked for train data
#null values in Age, Fare, Cabin.  
#Need to replace null values with some placeholder
for dset in dataset:
    dset['Age'].fillna(dset['Age'].median(), inplace=True)
    #dset['Cabin'].fillna(dset['Cabin'].mode()[0], inplace=True)
    dset['Fare'].fillna(dset['Fare'].mode()[0], inplace=True)
    dset['Embarked'].fillna(dset['Embarked'].mode()[0], inplace=True)
    
    

#do feature scalling for test and train data.
#As we can see that there is saluation present in name, let's create a new feature Title from saluation present in name attribute
for dset in dataset:
    dset['Title'] = dset.Name.str.split(',', expand=True)[1].str.split('.', expand=True)[0]
    
    #We can see that some of the passenger assigned with cabins whereas other don't have cabin assign to them
    #So here cabin could be an important point to find the survival of the passenger
    #Let's create one more attribute to check the existence of the cabin for a given passenger
    dset['Cabin_assigned'] = ~dset.Cabin.isnull()
    
    #Put some of the attributes like Age and Fare in categorial fashion
    dset['Age_Cat'] = pd.qcut(dset.Age, q = 4, labels = False)
    dset['Fare_Cat'] = pd.qcut(dset.Fare, q = 4, labels = False)
    
    #Create familly size which includes the passenger spouse, children and his/her parent in it.
    dset['Familly_Size'] = dset.SibSp + dset.Parch + 1
    
    

#Plot the title with data 
sns.countplot(x='Title', data = ds_train)

ds_train['Title'].replace({'Mlle':'Miss', 'Mme': 'Mrs', 'Ms':'Miss'}, inplace=True, regex=True)
ds_train['Title'].replace(['Don','Dona', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col', 'Capt', 'the Countess', 'Jonkheer'], 'Special', inplace=True, regex=True)
ds_test['Title'].replace({'Mlle':'Miss', 'Mme': 'Mrs', 'Ms':'Miss'}, inplace=True, regex=True)
ds_test['Title'].replace(['Don','Dona', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col', 'Capt', 'the Countess', 'Jonkheer', 'Speciala'], 'Special', inplace=True, regex=True)


#import required libraries for accuracy matrix calcuation
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, roc_curve

"""
 Find the accuracy of the model
"""
def accuracy_score_method(y_test, y_predicted):
    cm = confusion_matrix(y_test, y_predicted)
    acc_socre = accuracy_score(y_test, y_predicted)
    prec_score = precision_score(y_test, y_predicted)
    roc_curve_value = roc_curve(y_test, y_predicted) 
    print("Confusion matrix %s", cm)
    print("accuracy score %s", acc_socre)
    print("precision score %s", prec_score)
    print("roc curve %s", roc_curve_value)
    
#Since there are some of the column which we don't need now like Name, Cabin, Age and Fare.
#The reason being we have derived new attributes from the above attributes
remove_attribute = ['Age', 'Fare', 'Cabin', 'Name', 'PassengerId', 'Ticket', 'SibSp', 'Parch']
ds_train = ds_train.drop(remove_attribute, axis = 1)
ds_test = ds_test.drop(remove_attribute, axis = 1)

#Transforming data into binary variables
ds_train = pd.get_dummies(ds_train, drop_first=True)
X = ds_train.iloc[:,1:]
y = ds_train.iloc[:,0]

ds_test = pd.get_dummies(ds_test, drop_first=True)

#Split dataset into train and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)



#Create different models and check the result one by one.
#Start with Support Vector Machine SVC
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0, gamma=0.1)
classifier.fit(X_train, y_train)
svc_prediction = classifier.predict(X_test)
accuracy_score_method(y_test, svc_prediction)

#GridSearch algo for parameter tunning for SVC
 #from sklearn.grid_search import GridSearchCV
 #svc_parameter = [ {"kernel":['rbf']}, {"gamma":[1e-1, 1e-2]}]
 #gridsearch = GridSearchCV(  classifier, param_grid = svc_parameter, cv=10)
 #gridsearch.fit(X_train, y_train)
 #print("Best parameters %s", gridsearch.best_params_)
 #print("Best score %s", gridsearch.best_score_)


#Compute with RandomForest Tree classification
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier( n_estimators = 10, max_depth=6)
rf_classifier.fit(X_train, y_train)
rf_prediction = rf_classifier.predict(X_test)
accuracy_score_method(y_test, rf_prediction)

#GridSearch algo for parameter tunning for RandomeForest
#rf_parameter = [ {"n_estimators":[10,100,1000]}, {"max_depth":[1,3,5,6,7,10,100]}]
#gridsearch = GridSearchCV(  rf_classifier, param_grid = rf_parameter, cv=10)
#gridsearch.fit(X_train, y_train)
#print("Best parameters %s", gridsearch.best_params_)
#print("Best score %s", gridsearch.best_score_)

#LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
logistic_regression = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=0)
logistic_regression.fit(X_train, y_train)
logistic_prediction = logistic_regression.predict(X_test)
accuracy_score_method(y_test, logistic_prediction)

from sklearn.linear_model import SGDClassifier
sgdClassifier = SGDClassifier(alpha=0.01, epsilon=0.01, penalty='elasticnet')
sgdClassifier.fit(X_train, y_train)
sgd_prediction = sgdClassifier.predict(X_test)
accuracy_score_method(y_test, sgd_prediction)


voting_classifier = VotingClassifier(estimators=[('lr', logistic_regression), ('rf', rf_classifier), ('svc', classifier), ('sgd', sgdClassifier)], voting='hard')
voting_classifier.fit(X_train, y_train)
voting_prediction = voting_classifier.predict(X_test)
accuracy_score_method(y_test, voting_prediction)


from sklearn.model_selection import cross_val_score
accuriacy = cross_val_score(    estimator = voting_classifier, X= X_train, y=y_train, cv=10)
accuriacy.mean()







#Predict the final result to be submitted to Kaggal
ds_predict = voting_classifier.predict(ds_test)    

#prepare the submission.csv for kaggle
ds_test_copy['Survived'] = ds_predict
ds_test_copy[['PassengerId', 'Survived']].to_csv('MISSING_DATA_WITH_ENSEMBLE_SUBMISSION6.csv', index=False)

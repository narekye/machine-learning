#!/usr/bin/python
""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
sys.path.append("../udacity/tools/")
from email_preprocess import preprocess
from clear import clear


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###

from sklearn.svm import SVC # Support Vector Classifier
from sklearn import metrics

clear()

print("-- start")

classifier = SVC(kernel="rbf", C=10000.0)

#features_train = features_train[:int((len(features_train)/100))] 
#labels_train = labels_train[:int(len(labels_train)/100)] 

classifier.fit(features_train, labels_train)

prediction = classifier.predict(features_test)

accuracy = metrics.accuracy_score(labels_test, prediction) # 0.98 | 0.88 (with div by 100) | 0.61 wtih rbf
#########################################################
# All data with divided by 100
# rbf & C = 1.0 -> 0.61 accuracy
# rbf & C = 10.0 -> 0.61 
# rbf & C = 1000.0 -> 0.82
# rbf & C = 10000.0 -> 0.89
#########################################################

# rbf & C = 10000.0 -> 0.99
print(accuracy)

#########################################################



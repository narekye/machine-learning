#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../udacity/tools/")
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###

from sklearn.naive_bayes import GaussianNB
sys.path.append("../udacity/choose_your_own")
from class_vis import prettyPicture
from sklearn import metrics
pictureName = "naive_bayes_example.jpg"

classifier = GaussianNB()

classifier.fit(features_train, labels_train)

prediction = classifier.predict(features_test)

accuracy = metrics.accuracy_score(labels_test, prediction)

print(accuracy) 

# prettyPicture(classifier, features_test, labels_test, pictureName)


#########################################################



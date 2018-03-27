
#!/usr/bin/python
""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../udacity/tools/")
sys.path.append("../udacity/choose_your_own/")
# from email_preprocess import preprocess
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from clear import clear
from class_vis import show_img

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
# features_train, features_test, labels_train, labels_test = preprocess()
features_train, labels_train, features_test, labels_test = makeTerrainData()


#########################################################
### your code goes here ###
from sklearn.tree import tree
from sklearn.metrics import accuracy_score

clear()

print("Start execution")

# min_samples_split = 50
classifier = tree.DecisionTreeClassifier(min_samples_split=50)
classifier.fit(features_train, labels_train)

prediction = classifier.predict(features_test)

pictureName = "decision_tree_classifier_bigger.png"

accuracy = accuracy_score(labels_test ,prediction)

print(accuracy)

prettyPicture(classifier, features_test, labels_test, pictureName)

show_img(pictureName)

#########################################################
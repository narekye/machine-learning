
import sys
sys.path.append("../udacity/choose_your_own/")
sys.path.append("../udacity/tools/")
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, show_img
from clear import clear

features_train, labels_train, features_test, labels_test = makeTerrainData()

#########################################################
### your code goes here ###

from sklearn.svm import SVC
from sklearn import metrics

clear()

# C makes more training points correct 
# Gamma makes decision boundaries much closer or much far

classifier = SVC(kernel="rbf",C=10000) 
classifier.fit(features_train, labels_train)

prediction = classifier.predict(features_test)

accuracy = metrics.accuracy_score(labels_test, prediction)

print(accuracy)

pictureName = "svm_rbf.png"

prettyPicture(classifier, features_test, labels_test, pictureName)

show_img(pictureName)
#########################################################
import sys
sys.path.append("../udacity/choose_your_own/")
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()

#########################################################
### your code goes here ###

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

pictureName = "naive_bayes_example.jpg"

classifier = GaussianNB()   
classifier.fit(features_train, labels_train)

prettyPicture(classifier, features_test, labels_test, pictureName)

prediction = classifier.predict(features_test)

accuracy = metrics.accuracy_score(labels_test, prediction)

print(accuracy)

#########################################################
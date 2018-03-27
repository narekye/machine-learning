    
import sys
sys.path.append("../udacity/tools/")
sys.path.append("../udacity/choose_your_own")
from email_preprocess import preprocess
from class_vis import prettyPicture, show_img

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###

from sklearn.svm import SVC # Support Vector Classifier / Machine
from sklearn import metrics

classifier = SVC(kernel="rbf")
classifier.fit(features_train, labels_train)

prediction = classifier.predict(features_test)

accuracy = metrics.accuracy_score(labels_test, prediction) # 0.98

print(accuracy)

#########################################################
import math

from IPython import display

from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from train_model import train_model

from preprocess_features import preprocess_features
from preprocess_targets import preprocess_targets

from sklearn import metrics

import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

training_examples = preprocess_features(california_housing_dataframe.head(12000))
# training_examples.describe()

training_targets = preprocess_targets(california_housing_dataframe.head(12000))
# training_targets.describe()

validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
# validation_examples.describe()

validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
# validation_targets.describe() 

#plt.figure(figsize=(13, 8))

#ax = plt.subplot(1, 2, 1)
#ax.set_title("Validation Data")

#ax.set_autoscaley_on(False)
#ax.set_ylim([32, 43])
#ax.set_autoscalex_on(False)
#ax.set_xlim([-126, -112])
#plt.scatter(validation_examples["longitude"],
#            validation_examples["latitude"],
#            cmap="coolwarm",
#            c=validation_targets["median_house_value"] / validation_targets["median_house_value"].max())

#ax = plt.subplot(1,2,2)
#ax.set_title("Training Data")

#ax.set_autoscaley_on(False)
#ax.set_ylim([32, 43])
#ax.set_autoscalex_on(False)
#ax.set_xlim([-126, -112])
#plt.scatter(training_examples["longitude"],
#            training_examples["latitude"],
#            cmap="coolwarm",
#            c=training_targets["median_house_value"] / training_targets["median_house_value"].max())
#_ = plt.plot()
# plt.show()

linear_regressor = train_model(
    learning_rate=0.00003,
    steps=500,
    batch_size=5,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
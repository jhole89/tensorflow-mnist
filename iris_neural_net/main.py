import os
import urllib.request

import tensorflow as tf
import numpy as np


def get_train_inputs():
    """defines the test inputs"""
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)

    return x, y


def get_test_inputs():
    """define test inputs"""
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)

    return x, y


def new_samples():
    return np.array(
        [[6.4, 3.2, 4.5, 1.5],
         [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)


IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

# check for data and import if not available
if not os.path.exists(IRIS_TRAINING):
    raw = urllib.request.urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING, 'w') as f:
        f.write(raw)

if not os.path.exists(IRIS_TEST):
    raw = urllib.request.urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST, 'w') as f:
        f.write(raw)

# load data into tensorflow datasets
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)

# construct classifier
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="tmp/iris_model")

# fit model
classifier.fit(input_fn=get_train_inputs, steps=2000)

# evaluate accuracy
accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]
print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

predictions = list(classifier.predict(input_fn=new_samples))
print("New Samples, Class Predictions:  {}\n".format(predictions))

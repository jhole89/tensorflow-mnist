import os, shutil
import tensorflow as tf
import numpy as np
from urllib.request import urlopen


def clean_run(model_dir='', train_data='', test_data=''):
    if model_dir:
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
            print("\nCleaned: Model directory.\n")

    if train_data:
        if os.path.exists(train_data):
            os.remove(train_data)
            print("\nCleaned: Training data.\n")

    if test_data:
        if os.path.exists(test_data):
            os.remove(test_data)
            print("\nCleaned: Test data.\n")


def source_data(data):
    """download data if not present on local FS"""
    data_url = "http://download.tensorflow.org/data/" + data

    if not os.path.exists(data):
        raw = urlopen(data_url).read()
        with open(data, 'wb') as f:
            f.write(raw)

    return load_data(data)


def load_data(data):
    """load data into tensorflow datasets"""

    return tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=data,
        target_dtype=np.int,
        features_dtype=np.float32)


def get_inputs(dataset):
    """defines the inputs"""
    x = tf.constant(dataset.data)
    y = tf.constant(dataset.target)

    return x, y


def new_samples():
    return np.array(
        [[6.4, 3.2, 4.5, 1.5],
         [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)


def main():

    tf.logging.set_verbosity(tf.logging.INFO)

    model_dir = 'iris_model'
    IRIS_TRAINING = "iris_training.csv"
    IRIS_TEST = "iris_test.csv"

    clean_run(model_dir=model_dir)
    training_set = source_data(IRIS_TRAINING)
    test_set = source_data(IRIS_TEST)

    validation_metrics = {
        "accuracy":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_accuracy,
                prediction_key="classes"),
        "precision":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_precision,
                prediction_key="classes"),
        "recall":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_recall,
                prediction_key="classes")
    }

    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        test_set.data,
        test_set.target,
        every_n_steps=50,
        metrics=validation_metrics,
        early_stopping_metric="loss",
        early_stopping_metric_minimize=True,
        early_stopping_rounds=200)

    # construct classifier
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 20, 10],
        n_classes=3,
        model_dir=model_dir,
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))

    # fit model
    classifier.fit(input_fn=lambda: get_inputs(training_set),
                   steps=2000,
                   monitors=[validation_monitor])

    # evaluate accuracy
    accuracy_score = classifier.evaluate(input_fn=lambda: get_inputs(test_set), steps=1)["accuracy"]
    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

    predictions = list(classifier.predict(input_fn=new_samples))
    print("New Samples, Class Predictions:  {}\n".format(predictions))


if __name__ == '__main__':
    main()

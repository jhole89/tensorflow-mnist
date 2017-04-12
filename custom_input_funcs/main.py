import os, itertools
import tensorflow as tf
import pandas as pd

from urllib.request import urlopen


def source_data(data):
    data_url = "http://download.tensorflow.org/data/" + data

    if not os.path.exists(data):
        raw = urlopen(data_url).read()
        with open(data, 'wb') as f:
            f.write(raw)

    return load_data_to_pandas(data)


def load_data_to_pandas(data):
    columns = ["crim", "zn", "indus", "nox", "rm",
               "age", "dis", "tax", "ptratio", "medv"]
    return pd.read_csv(data, skipinitialspace=True,
                       skiprows=1, names=columns)


def input_fn(data_set):
    feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
    labels = tf.constant(data_set[LABEL].values)
    return feature_cols, labels


if __name__ == "__main__":

    BOSTON_TRAIN = "boston_train.csv"
    BOSTON_TEST = "boston_test.csv"
    BOSTON_PREDICT = "boston_predict.csv"

    training_set = source_data(BOSTON_TRAIN)
    test_set = source_data(BOSTON_TEST)
    prediction_set = source_data(BOSTON_PREDICT)

    tf.logging.set_verbosity(tf.logging.INFO)

    FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
    LABEL = "medv"

    feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

    regressor = tf.contrib.learn.DNNRegressor(
        feature_columns=feature_cols,
        hidden_units=[10, 10],
        model_dir="tmp/boston_model")

    regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)

    evaluate = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)

    loss_score = evaluate["loss"]
    print("Loss: {0:f}".format(loss_score))

    predict = regressor.predict(input_fn=lambda: input_fn(prediction_set))
    predictions = list(itertools.islice(predict, 6))
    print("Predictions: {}".format(str(predictions)))
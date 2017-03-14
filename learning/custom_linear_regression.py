import numpy as np
import tensorflow as tf


def model(features, labels, mode):
    # build model
    W = tf.get_variable("W", [1], dtype=tf.float64)
    b = tf.get_variable("d", [1], dtype=tf.float64)
    y = W * features['x'] + b

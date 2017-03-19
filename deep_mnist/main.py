import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x_patch = 5
y_patch = 5
input_channels = 1
features = 32

W_conv1 = weight_variable([x_patch, y_patch, input_channels, features])
b_conv1 = bias_variable([features])

x_pixels = 28
y_pixels = 28

x_image = tf.reshape(x, [-1, x_pixels, y_pixels, input_channels])
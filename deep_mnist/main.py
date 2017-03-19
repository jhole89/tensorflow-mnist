import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# import data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# set model values
num_pixels = 784
num_images = None
num_scenarios = 10

# build model
images = tf.placeholder(tf.float32, [num_images, num_pixels])
weights = tf.Variable(tf.zeros([num_pixels, num_scenarios]))
bias = tf.Variable(tf.zeros([num_scenarios]))
estimates = tf.nn.softmax(tf.matmul(images, weights) + bias)


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


def build_layer(channels, features):
    x_patch = 5
    y_patch = 5
    weight_conv = weight_variable([x_patch, y_patch, channels, features])
    bias_conv = bias_variable([features])
    return weight_conv, bias_conv


# first layer
conv1_channel = 1
conv1_features = 32

weight_conv1, bias_conv1 = build_layer(conv1_channel, conv1_features)

image_width = 28
image_height = 28

x_image = tf.reshape(images, [-1, image_width, image_height, conv1_channel])
h_conv1 = tf.nn.relu(conv2d(x_image, weight_conv1) + bias_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second layer
conv2_features = 64

weight_conv2, bias_conv2 = build_layer(conv1_features, conv2_features)


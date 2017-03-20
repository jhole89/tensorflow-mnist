import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
sess = tf.InteractiveSession()

# import data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# set model values
num_pixels = 784
num_images = None
num_scenarios = 10

# build model
images = tf.placeholder(tf.float32, [num_images, num_pixels])
y_ = tf.placeholder(tf.float32, shape=[num_images, num_scenarios])
weights = tf.Variable(tf.zeros([num_pixels, num_scenarios]))
bias = tf.Variable(tf.zeros([num_scenarios]))


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
h_conv2 = tf.nn.relu(conv2d(h_pool1, weight_conv2) + bias_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# connected layer
weight_connected_1 = weight_variable([7 * 7 * conv2_features, 1024])
bias_connected_1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_connected_1 = tf.nn.relu(tf.matmul(h_pool2_flat, weight_connected_1) + bias_connected_1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_connected1_drop = tf.nn.dropout(h_connected_1, keep_prob)

# readout layer
weight_connected2 = weight_variable([1024, 10])
bias_connected2 = bias_variable([10])

y_conv = tf.matmul(h_connected1_drop, weight_connected2) + bias_connected2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        images:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={images: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    images: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
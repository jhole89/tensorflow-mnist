import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(images, weights):
    return tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(images):
    return tf.nn.max_pool(images, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def build_layer(layer_count, input_channels, output_channels, image_conv):
    patch_width = 5
    patch_height = 5

    weight_conv = weight_variable([patch_width, patch_height, input_channels, output_channels])
    bias_conv = bias_variable([output_channels])

    hidden_conv = tf.nn.relu(conv2d(image_conv, weight_conv) + bias_conv)
    hidden_pool = max_pool_2x2(hidden_conv)

    layer_count += 1
    return layer_count, hidden_pool


sess = tf.InteractiveSession()

# import data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# set model values
num_pixels = 784
num_images = None
num_scenarios = 10

# define model parameters
images = tf.placeholder(tf.float32, [num_images, num_pixels])
predictions = tf.placeholder(tf.float32, shape=[num_images, num_scenarios])
weights = tf.Variable(tf.zeros([num_pixels, num_scenarios]))
bias = tf.Variable(tf.zeros([num_scenarios]))

# reshape images to a 4d tensor
image_width = 28
image_height = 28
colour_channels = 1

reshaped_images = tf.reshape(images, [-1, image_width, image_height, colour_channels])


# first layer
layer1_input_channels = 1
layer1_output_channels = 32
layer_count, hidden_pool1 = build_layer(0, layer1_input_channels, layer1_output_channels, reshaped_images)


# second layer
layer2_output_channels = 2 * layer1_output_channels
layer_count, hidden_pool2 = build_layer(layer_count, layer1_output_channels, layer2_output_channels, hidden_pool1)


# connected layer
new_image_width = int(image_width / (2**layer_count))
new_image_height = int(image_height / (2**layer_count))
neurons = 1024

weight_connected = weight_variable([new_image_width * new_image_height * layer2_output_channels, neurons])
bias_connected = bias_variable([neurons])

hidden_pool2_flat = tf.reshape(hidden_pool2, [-1, new_image_width * new_image_height * layer2_output_channels])
hidden_connected = tf.nn.relu(tf.matmul(hidden_pool2_flat, weight_connected) + bias_connected)


# dropout
keep_prob = tf.placeholder(tf.float32)
hidden_connected_dropout = tf.nn.dropout(hidden_connected, keep_prob)

# readout layer
readout_weight = weight_variable([neurons, num_scenarios])
readout_bias = bias_variable([num_scenarios])

prediction_conv = tf.matmul(hidden_connected_dropout, readout_weight) + readout_bias

# train model
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=predictions, logits=prediction_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(prediction_conv, 1), tf.argmax(predictions, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

for i in range(20000):
    batch = mnist.train.next_batch(50)

    if i % 100 == 0:
        train_accuracy = accuracy.eval(
            feed_dict={images: batch[0], predictions: batch[1], keep_prob: 1.0})

        print("step %d, training accuracy %g" % (i, train_accuracy))

    train_step.run(feed_dict={images: batch[0], predictions: batch[1], keep_prob: 0.5})

# evaluate model
print("test accuracy %g" % accuracy.eval(
    feed_dict={images: mnist.test.images, predictions: mnist.test.labels, keep_prob: 1.0}))

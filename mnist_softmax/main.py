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

# define loss
values = tf.placeholder(tf.float32, [num_images, num_scenarios])

cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(values * tf.log(estimates), reduction_indices=[1]))

# set training
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# initialise
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# do training
for step in range(1000):
    batch_images, batch_values = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={images: batch_images, values: batch_values})

# evaluate
correct_prediction = tf.equal(tf.argmax(estimates, 1), tf.argmax(values, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={images: mnist.test.images, values: mnist.test.labels}))

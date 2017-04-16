import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# config
batch_size = 100
learning_rate = 0.5
training_epochs = 5
log_path = 'tmp/tensorflow/mnist/logs/mnist_with_summaries'
x_pixels = 28
y_pixels = 28
image_pixels = x_pixels * y_pixels
num_images = None
num_scenarios = 10
colour_channels = 1

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.name_scope('input'):

    images = tf.placeholder(tf.float32, shape=[num_images, image_pixels], name='image-input')
    labels = tf.placeholder(tf.float32, shape=[num_images, num_scenarios], name='label-input')

with tf.name_scope('weights'):
    weight = tf.Variable(tf.zeros([image_pixels, num_scenarios]))

with tf.name_scope('biases'):
    bias = tf.Variable(tf.zeros([num_scenarios]))

with tf.name_scope('softmax'):
    model = tf.nn.softmax(tf.matmul(images, weight) + bias)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(model), reduction_indices=[1]))

with tf.name_scope('train'):
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('cost', cross_entropy)
tf.summary.scalar('accuracy', accuracy)

summary_op = tf.summary.merge_all()

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

writer = tf.summary.FileWriter(log_path,graph=tf.get_default_graph())

for epoch in range(training_epochs):
    batch_count = int(mnist.train.num_examples/batch_size)

    for i in range(batch_count):
        batch_image, batch_label = mnist.train.next_batch(batch_size)

        _, summary = sess.run([train_op, summary_op],
                              feed_dict={images: batch_image, labels: batch_label})

        writer.add_summary(summary, epoch * batch_count + i)

    if epoch % 5 == 0:
        print("Epoch: ", epoch)

print("Accuracy: ", sess.run(accuracy, feed_dict={images: mnist.test.images,
                                                  labels: mnist.test.labels}))

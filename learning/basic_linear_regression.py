import tensorflow as tf

# Model params
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W * x + b

# Loss
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# setup
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# training loop
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

# evaluate
current_W, current_b, current_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s" %(current_W, current_b, current_loss))

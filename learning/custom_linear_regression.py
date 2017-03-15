import numpy as np
import tensorflow as tf


def model(features, labels, mode):
    # build model
    W = tf.get_variable("W", [1], dtype=tf.float64)
    b = tf.get_variable("d", [1], dtype=tf.float64)
    y = W * features['x'] + b

    # loss graph
    loss = tf.reduce_sum(tf.square(y - labels))

    # training graph
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    return tf.contrib.learn.ModelFnOps(mode=mode, predictions=y, loss=loss, train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)

# define data set
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_func = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, 4, num_epochs=1000)

# train
estimator.fit(input_fn=input_func, steps=1000)

# evaluate
print(estimator.evaluate(input_fn=input_func, steps=10))

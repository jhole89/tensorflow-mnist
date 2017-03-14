import tensorflow as tf
import numpy as np

# declare list of features and network depth 1
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

# use predefined Linear Regression model
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# training data
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])

input_func = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, batch_size=4, num_epochs=1000)

estimator.fit(input_fn=input_func, steps=1000)

print(estimator.evaluate(input_fn=input_func))

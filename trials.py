import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[no.ones((m, 1)), housing.data]

x = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="x")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(x)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, x)), XT), y)

with tf.Session() as sess:
   theta_value = theta.eval()

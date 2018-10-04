import tensorflow as tf
from sklearn.datasets import fetch_california_housing

dataset = fetch_california_housing()
x = tf.constant(dataset.data)
y = tf.constant(dataset.feature)
xt = tf.matrix_transpose(x)

theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(xt, x)), xt), y);

init = tf.global_variables_initializer()
with tf.Session as sess:
   init.run()
   theta_value = theta.eval()

import tensorflow as tf

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2
init = tf.global_variables_initializer()
with tf.Session() as sess:
  init.run()
  result = f.eval()
print(result)

x = tf.Variable(2, name="x")
f = x*x*y + y + 2
init = tf.global_variables_initializer()

with tf.Session() as sess:
  init.run()
  result = sess.run(f)
print(result)

sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
sess.close()
print(result)

x2 = tf.Variable(5, name="x2")
print (x2.graph is tf.get_default_graph())

w = tf.constant(2)
t = w + 3
z = t + 2
with tf.Session() as sess:
   print(sess.run(z))
   print(sess.run(t)) # t evaluated twice
with tf.Session() as sess:
   print(sess.run([z, t]))

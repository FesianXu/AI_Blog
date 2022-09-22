import tensorflow as tf

X1 = tf.placeholder(tf.float32, shape=(1,), name="X1")
X2 = tf.placeholder(tf.float32, shape=(1,), name="X2")

h1 = tf.multiply(X1, X2)
h2 = tf.add(h1, X1)
output = tf.div(h2, X2)
print_op = tf.Print(h1, [h1])

grad = tf.gradients(output, [X1, X2])

feed_dict = {
    "X1": 0.6, "X2": 0.2
}
sess = tf.Session()
output_v = sess.run(output, feed_dict)
grad_v = sess.run(grad, feed_dict)
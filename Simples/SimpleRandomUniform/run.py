from __future__ import division, absolute_import, print_function
import numpy as np
import tensorflow as tf


data = np.array([[1, 2], [3, 4]], dtype="Float32")
inputs = tf.placeholder(tf.float32, (2, 2), name="input")
x = tf.random_uniform([2, 2])
output = inputs + x

with tf.Session() as sess:
    print(sess.run(output, feed_dict={inputs:data}))
    tf.train.write_graph(tf.get_default_graph(), "my-model", "SimpleRandomUniform.pb", as_text=False)

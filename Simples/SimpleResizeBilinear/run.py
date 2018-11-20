from __future__ import absolute_import, print_function, division
import tensorflow as tf

a = tf.placeholder(tf.float32, shape=(1,3,3,3), name="input")

y = tf.image.resize_bilinear(a, [6,6], False, "output")

with tf.Session() as sess:
        print(sess.run(y, feed_dict={a: [[ [[1,2,3],[4,5,6],[7,8,9]], [[1,2,3],[4,5,6],[7,8,9]], [[1,2,3],[4,5,6],[7,8,9]] ]]} ))
        tf.train.write_graph(tf.get_default_graph(), "./my-model/", "SimpleResizeBilinear.pb", as_text=False)

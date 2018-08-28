from __future__ import division, print_function, absolute_import                                                                                                                             
import tensorflow as tf
import numpy as np

inputs = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype='Float32')
print(inputs)
print(inputs.shape)

matrix = np.ones((27, 6), dtype='Float32')
print(matrix)
print(matrix.shape)

bias = np.ones((6), dtype='Float32') * 5
print(bias)
print(bias.shape)


tf_inputs = tf.placeholder(tf.float32, (1, 3, 3, 3), 'input')

tf_reshape = tf.reshape(tf_inputs, [1, -1])
print(tf_reshape)
print(tf_reshape.shape)

tf_matmul = tf.matmul(tf_reshape, matrix)
print(tf_matmul)
print(tf_matmul.shape)

tf_biasadd = tf.nn.bias_add(tf_matmul, bias)
print(tf_biasadd)
print(tf_biasadd.shape)

tf_relu = tf.nn.relu(tf_biasadd, 'output')
print(tf_relu)
print(tf_relu.shape)

with tf.Session() as sess:
  print(sess.run(tf_relu, feed_dict={tf_inputs:inputs}))
  tf.train.write_graph(tf.get_default_graph(), "./my-model/", "SimpleFullyConnected.pb", as_text=False)

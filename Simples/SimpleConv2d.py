from __future__ import division, absolute_import, print_function                                                                                                                             
import tensorflow as tf
import numpy as np

print(tf.__version__)

inputs = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype='Float32')
print(input)
filter = np.array([[[[1], [1], [1]], [[1], [1], [1]]], [[[1], [1], [1]], [[1], [1], [1]]]], dtype='Float32')
print(filter)
bias = np.array([5], dtype='Float32')

conv2d_inputs = tf.placeholder(tf.float32, (1, 3, 3, 3), "inputs")
conv2d_filter = tf.constant(filter, tf.float32, (2, 2, 3, 1), "filter")
conv2d_stride = [1, 1, 1, 1]

conv2d_filter_output = tf.nn.conv2d(conv2d_inputs, conv2d_filter, conv2d_stride, "VALID", True, 'NHWC', 'conv2d_filter')
conv2d_bias_output = tf.nn.bias_add(conv2d_filter_output, bias, 'NHWC', 'conv2d_bias')
conv2d_relu_output = tf.nn.relu(conv2d_bias_output, 'output')

#x = tf.get_variable("x", [1])

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  print(sess.run(conv2d_relu_output, feed_dict={conv2d_inputs: inputs}))
  tf.train.write_graph(tf.get_default_graph(), "./my-model/", "my-model.pb", as_text=False)

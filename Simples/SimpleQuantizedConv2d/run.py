from __future__ import division, print_function, absolute_import
import tensorflow as tf

inputs_data = [[[[0.1], [1.1], [2.2], [3.3], [4.4]], [[5.6], [7.8], [8.9], [9.1], [1.1]], [[1.2], [2.3], [3.4], [4.5], [5.6]], [[6.7], [7.8], [8.9], [9.2], [2.21]], [[1.11], [2.22], [3.33], [4.44], [5.55]]]]

filters_data = [ [[[1.8]], [[1.9]], [[1.0]]], [[[1.1]], [[1.2]], [[1.3]]], [[[1.4]], [[1.5]], [[1.6]]] ]


inputs = tf.placeholder(tf.float32, (1, 5, 5, 1), "inputs")

filters = tf.placeholder(tf.float32, (3, 3, 1, 1), "filters")

quantized_inputs = tf.quantization.quantize(inputs, 0, 10, tf.quint8, 'MIN_COMBINED', 'HALF_AWAY_FROM_ZERO', 'quantized_inputs')

quantized_filters = tf.quantization.quantize(filters, 0, 2, tf.quint8, 'MIN_COMBINED', 'HALF_AWAY_FROM_ZERO', 'quantized_filters')


conv2d_output = tf.nn.quantized_conv2d(quantized_inputs.output, quantized_filters.output, 0, 10, 0, 2, [1, 1, 1, 1], "VALID", tf.qint32, [1, 1, 1, 1],'output')


conv2d_dequantized = tf.quantization.dequantize(conv2d_output.output, conv2d_output.min_output, conv2d_output.max_output, 'MIN_COMBINED', 'outputs_dequantized')


conv2d_filter_output = tf.nn.conv2d(inputs, filters, [1,1,1,1], "VALID", True, 'NHWC',[1,1,1,1], 'conv2d_filter')

with tf.Session() as sess:
    print(sess.run(quantized_inputs, feed_dict={inputs: inputs_data}))
    print(sess.run(quantized_filters, feed_dict={filters: filters_data}))
    print(sess.run(conv2d_output, feed_dict={inputs:inputs_data, filters:filters_data}))
    print(sess.run(conv2d_dequantized, feed_dict={inputs:inputs_data, filters:filters_data}))
    print(sess.run(conv2d_filter_output, feed_dict={inputs:inputs_data, filters:filters_data}))

from __future__ import division, print_function, absolute_import
import tensorflow as tf

inputs_data = [0.1, 1.1, 2.2, 3.3, 4.4]

inputs = tf.placeholder(tf.float32, (5), "inputs")

quantized = tf.quantization.quantize(inputs, 0, 255, tf.quint8, 'MIN_COMBINED', 'HALF_AWAY_FROM_ZERO', 'outputs_quantized')

dequantized = tf.quantization.dequantize(quantized.output, 0, 255, 'MIN_COMBINED', 'outputs_dequantized')


with tf.Session() as sess:
    print(sess.run(quantized, feed_dict={inputs: inputs_data}))
    print(sess.run(dequantized, feed_dict={inputs: inputs_data}))

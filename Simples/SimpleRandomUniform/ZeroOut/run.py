from __future__ import print_function, division, absolute_import
import tensorflow as tf
import numpy as npLog.d(TAG, "onCreate: outputs = " + outputs.getFloat(0));
zero_out_module = tf.load_op_library('zero_out.so')

data = np.array([[1, 2], [3, 4]], dtype="Int32")
inputs = tf.placeholder(tf.int32, (2, 2), name='input')
out = zero_out_module.zero_out(inputs, "zero_out")

with tf.Session() as sess:
    print(sess.run(out, feed_dict={inputs:data}))
    tf.train.write_graph(tf.get_default_graph(), "my-model", "SimpleZeroOut.pb", as_text=False)



from __future__ import absolute_import, division, print_function
import tensorflow as tf


''' This sample is implementation of a simple y = a + b, where a, b are input, y is output. '''

a = tf.placeholder(tf.float32, shape=(1), name="inputa")
b = tf.constant(2, dtype=tf.float32)

x = tf.get_variable("x", [1])

y = tf.add(a, b, "output")

init = tf.global_variables_initializer()

saver = tf.train.Saver({'x': x})

with tf.Session() as sess:
	sess.run(init)

	print(sess.run(y, feed_dict={a: [3]}))

	# Save model's variables to checkpoint
	saver.save(sess, './my-model/my-model', global_step=10)

	# Save model's computational graph to protobuf
	tf.train.write_graph(tf.get_default_graph(), "./my-model/", "my-model.pb", as_text=False)

	# Note. use tf.import_graph_def to restore the protobuf file.
	#tf.saved_model.simple_save(sess, "./saved_model/", inputs={"a": a}, outputs={"y": y})

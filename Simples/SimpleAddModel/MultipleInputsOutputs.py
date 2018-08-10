import tensorflow as tf                                                                                                                                                                      

input0 = tf.placeholder(tf.float32, (2), "input0")
input1 = tf.placeholder(tf.float32, (2), "input1")
input2 = tf.placeholder(tf.float32, (2), "input2")

inputs = tf.concat([input0, input1, input2], 0, "inputs")

factor = tf.constant(5, tf.float32, [1], "factor")

outputs = tf.multiply(inputs, factor, "outputs")

out0, out1, out2 = tf.split(outputs, 3)

output0 = tf.multiply(out0, factor, "output0")
output1 = tf.multiply(out1, factor, "output1")
output2 = tf.multiply(out2, factor, "output2")

x = tf.get_variable("x", [1])

saver = tf.train.Saver({'x': x})
init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  print(sess.run([output0, output1, output2], feed_dict={input0:[1.0, 2.0], input1:[3.0, 4.0], input2: [5.0, 6.0]}))
  print input0
  print inputs
  print outputs
  tf.train.write_graph(tf.get_default_graph(), "./my-model/", "my-model.pb", as_text=True)

  saver.save(sess, './my-model/my-model', global_step=1)


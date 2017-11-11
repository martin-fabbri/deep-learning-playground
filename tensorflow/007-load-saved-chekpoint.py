import tensorflow as tf

# save the file path and save the data
save_file = "./model.ckpt"

# tensorflow variables: weights and bias
weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

# class used to save and/restore tensorflow
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, save_file)

    print("Weights:")
    print(sess.run(weights))
    print("Bias:")
    print(sess.run(bias))

    saver.save(sess, save_file)

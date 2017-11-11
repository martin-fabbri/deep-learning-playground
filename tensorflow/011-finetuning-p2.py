import tensorflow as tf

save_file = 'model.ckpt'

# Two Tensor Variables: weights and bias
bias = tf.Variable(tf.truncated_normal([3]), name="bias_0")
weights = tf.Variable(tf.truncated_normal([2, 3]), name="weights_0")


saver = tf.train.Saver()

# Print the name of Weights and Bias
print('Save Weights: {}'.format(weights.name))
print('Save Bias: {}'.format(bias.name))

with tf.Session() as sess:
    saver.restore(sess, save_file)


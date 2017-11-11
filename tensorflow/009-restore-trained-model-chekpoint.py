import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.001
n_input = 784
n_classes = 10

# save the file path and save the data

mnist = input_data.read_data_sets(".", one_hot=True)

features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

# tensorflow variables: weights and bias
weights = tf.Variable(tf.truncated_normal([n_input, n_classes]))
bias = tf.Variable(tf.truncated_normal([n_classes]))

logits = tf.add(tf.matmul(features, weights), bias)

cost = tf.reduce_mean(\
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
    .minimize(cost)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# class used to save and/restore tensorflow
import math

save_file = './train_model.ckpt'
batch_size = 128
n_epochs = 100
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    saver.restore(sess, save_file)

    test_accuracy = sess.run(
        accuracy,
        feed_dict={features: mnist.test.images, labels: mnist.test.labels})

print('Test Accuracy: {}'.format(test_accuracy))
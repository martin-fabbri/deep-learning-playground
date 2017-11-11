from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

import tensorflow as tf

# parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 128
display_step = 1

n_input = 784
n_classes = 10

n_hidden_layer = 256

weights = {
    'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
}

biases = {
    'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# tf graph input
x = tf.placeholder(tf.int32, [None, 28, 28, 1])
y = tf.placeholder(tf.int32, [None, n_classes])
x_flat = tf.reshape(tf.cast(x, tf.float32), [-1, n_input])

# hidden layer with RELU activation
layer_1 = tf.add(tf.matmul(x_flat, weights['hidden_layer']), biases["hidden_layer"])
layer_1 = tf.nn.relu(layer_1)

# output layer with linear activation
logits = tf.add(tf.matmul(layer_1, weights["out"]), biases["out"])

# optimizer
# define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate the accuracy of the model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# session

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))


# define the input function for training


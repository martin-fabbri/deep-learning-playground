import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")

data = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("Size of:")
print(f"- Training set: \t{len(data.train.labels)}")
print(f"- Test-set: \t\t{len(data.test.labels)}")
print(f"- Validation-set: \t{len(data.validation.labels)}")

print("-- Test labels --")
print(data.test.labels[0:5, :])

data.test.cls = np.array([label.argmax() for label in data.test.labels])
data.train.cls = np.array([label.argmax() for label in data.train.labels])
print(f"data.test.cls[0:5]: {data.test.cls[0:5]}")

img_size = 28

# images are stored in a one-dimensional array of this length
img_size_flat = img_size * img_size

# tuple with height and width of images used to reshape arrays
img_shape = (img_size, img_size)


# number of classes, one class for each digit
num_classes = 10


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # plot image
        ax.imshow(images[i].reshape(img_shape), cmap="binary")

        if cls_pred is None:
            xlabel = f"True: {cls_true[i]}"
        else:
            xlabel = f"True: {cls_true[i]}, Pred: {cls_pred[i]}"

        ax.set_xlabel(xlabel)

        # remove ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])


# plot a few images to see if our data is correct
images = data.train.images[0:9]
cls_true = data.train.cls[0:9]
plot_images(images=images, cls_true=cls_true)

# A TensorFlow graph consists of the following parts:
# - Placeholder variables used to change the input to the graph
# - Variables are used to optimized the model
x = tf.placeholder(tf.float32, [None, img_size_flat])

y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))

logits = tf.nn.softmax(tf.matmul(x, weights) + biases)
y_pred = tf.nn.softmax(logits)

y_pred_cls = tf.argmax(y_pred, dimension=1)

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
)

optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

session = tf.InteractiveSession()
init = tf.global_variables_initializer()
session.run(init)

batch_size = 100
num_interactions = 10000

for _ in range(num_interactions):
    x_batch, y_true_batch = data.train.next_batch(batch_size)

    feed_dict_train = {x: x_batch, y_true: y_true_batch}

    session.run(optimizer, feed_dict=feed_dict_train)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(session.run(accuracy, feed_dict={x: data.test.images, y_true_cls: data.test.cls}))



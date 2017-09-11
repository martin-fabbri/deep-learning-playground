import time
import sys
import numpy as np
import math
from collections import Counter


class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes=10, learning_rate=0.1):
        """Create a SentimenNetwork with the given settings
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training

        """
        # Assign a seed to our random number generator to ensure we get
        # reproducable results during development
        np.random.seed(1)

        # process the reviews and their associated labels so that everything
        # is ready for training
        self.pre_process_data(reviews, labels)

        # Build the network to have the number of hidden nodes and the learning rate that
        # were passed into this initializer. Make the same number of input nodes as
        # there are vocabulary words and create a single output node.
        self.init_network(len(self.review_vocab), hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels):
        # populate review_vocab with all of the words in the given reviews
        review_vocab = {word for review in reviews for word in review.split(" ")}

        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)

        # populate label_vocab with all of the words in the given labels.
        label_vocab = {word for word in labels}

        # Convert the label vocabulary set to a list so we can access labels via indices
        self.label_vocab = list(label_vocab)

        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)

        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {word: index for index, word in enumerate(review_vocab)}

        # Create a dictionary of labels mapped to index positions
        self.label2index = {label: index for index, label in enumerate(label_vocab)}

    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Store the number of nodes in input, hidden, and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights

        # initialize self.weights_0_1 as a matrix of zeros.
        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))

        # initialize self.weights_1_2 as a matrix of random values.
        mu, sigma = 0, self.output_nodes ** -0.5  # mean and standard deviation
        self.weights_1_2 = np.random.normal(mu, sigma, (self.hidden_nodes, self.output_nodes))

        # Create the input layer.
        self.layer_0 = np.zeros((1, self.input_nodes))

    def update_input_layer(self, review):
        # clear out previous state by resetting the layer to be all 0s
        self.layer_0 *= 0

        # count how many times each word is used in the given review and store the results in layer_0
        review_word_count = Counter(review.split(" "))
        for word, count in review_word_count.items():
            if word in self.word2index:
                self.layer_0[:, self.word2index[word]] = 1

    def get_target_for_label(self, label):
        return 1 if label == "POSITIVE" else 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_output_2_derivative(self, output):
        return output * (1 - output)

    def train(self, training_reviews, training_labels):

        # make sure out we have a matching number of reviews and labels
        assert (len(training_reviews) == len(training_labels))

        # Keep track of correct predictions to display accuracy during training
        correct_so_far = 0

        # Remember when we started for printing time statistics
        start = time.time()

        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):
            # get the next review and its correct label
            review, label = training_reviews[i], training_labels[i]

            self.update_input_layer(review)
            layer_1 = self.layer_0.dot(self.weights_0_1)
            layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))

            # Output error
            layer_2_error = layer_2 - self.get_target_for_label(
                label)  # Output layer error is the difference between desired target and actual output.
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)

            # Backpropagated error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T)  # errors propagated to the hidden layer
            layer_1_delta = layer_1_error  # hidden layer gradients - no nonlinearity so it's the same as the error

            # Update the weights
            self.weights_1_2 -= layer_1.T.dot(
                layer_2_delta) * self.learning_rate  # update hidden-to-output weights with gradient descent step
            self.weights_0_1 -= self.layer_0.T.dot(
                layer_1_delta) * self.learning_rate  # update input-to-hidden weights with gradient descent step

            # TODO: Keep track of correct predictions. To determine if the prediction was
            #       correct, check that the absolute value of the output error
            #       is less than 0.5. If so, add one to the correct_so_far count.
            if (layer_2 >= 0.5 and label == 'POSITIVE') or (layer_2 < 0.5 and label == 'NEGATIVE'):
                correct_so_far += 1

            # For debug purposes, print out our prediction accuracy and speed
            # throughout the training process.

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i / float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i + 1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i + 1))[:4] + "%")
            if i % 2500 == 0:
                print("")

    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """

        # keep track of how many correct predictions we make
        correct = 0

        # we'll time how many predictions per second we make
        start = time.time()

        # Loop through each of the given reviews and call run to predict
        # its label.
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if pred == testing_labels[i]:
                correct += 1

            # For debug purposes, print out our prediction accuracy and speed
            # throughout the prediction process.

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i / float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i + 1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i + 1))[:4] + "%")

    def run(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        self.update_input_layer(review.lower())

        layer_1 = self.layer_0.dot(self.weights_0_1)
        layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))

        if layer_2[0] >= 0.5:
            return "POSITIVE"
        else:
            return "NEGATIVE"


if __name__ == "__main__":
    g = open('reviews.txt', 'r')  # What we know!
    reviews = list(map(lambda x: x[:-1], g.readlines()))
    g.close()

    g = open('labels.txt', 'r')  # What we WANT to know!
    labels = list(map(lambda x: x[:-1].upper(), g.readlines()))
    g.close()

    print(f"Number of reviews: {len(reviews)}")

    mlp = SentimentNetwork(reviews[:-1000], labels[:-1000], learning_rate=0.01)
    mlp.update_input_layer(reviews[1])
    print("Pause here to analise results. Set a breakpoint!")
    mlp.train(reviews[:-1000], labels[:-1000])

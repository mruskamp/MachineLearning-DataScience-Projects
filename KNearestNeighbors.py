import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

def display_digit(digit):
    plt.imshow(digit.reshape(28,28), cmap="Greys", interpolation="nearest")
    plt.show()

def get_majority_predicted_label(labels, indices):
    predicted_labels = []
    for i in indices:
        predicted_labels.append(labels[i])
    predicted_labels = np.array(predicted_labels)
    print(predicted_labels)

    counts = np.bincount(predicted_labels)
    return np.argmax(counts)

mnist = input_data.read_data_sets("MNIST_data/")

training_digits, training_labels = mnist.train.next_batch(4000)
test_digits, test_labels = mnist.train.next_batch(200);

display_digit(training_digits[1])


tf.reset_default_graph()
training_digits_ph = tf.placeholder("float", [None, 784])
test_digit_ph = tf.placeholder("float", [784])

L1_dist = tf.abs(tf.subtract(training_digits_ph, test_digit_ph))
L1_distance = tf.reduce_sum(L1_dist, axis=1)
pred_knn_L1 = tf.nn.top_k(tf.negative(L1_distance), k=5)
accuracy = 0.0

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(len(test_digits)):
        _, indices = sess.run(pred_knn_L1, feed_dict={training_digits_ph: training_digits, test_digit_ph: test_digits[i,:]})

        predicted_label = get_majority_predicted_label(training_labels, indices)
        print("Test:", i, "Prediction:", predicted_label, "True Label:", test_labels[i])

        if(predicted_label == test_labels[i]):
            accuracy += 1.0/len(test_digits)
    print("Accuracy:", accuracy)









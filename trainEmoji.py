# _*_ coding:utf-8 _*_
import numpy as np
from utils import *
import emoji
import os
import tensorflow as tf
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def predict(X, Y, W, b, word_to_vec_map):
    m = Y.shape[0] # number of training examples
    a = []
    for i in range(m): # Loop over the training examples
        # Average the word vectors of the words from the i'th training example
        avg = sentence_to_avg(X[i], word_to_vec_map)
        # Forward propagate the avg through the softmax layer
        z = np.dot(W, avg) + b
        l = list(softmax(z))
        max_value = max(l)
        a.append(l.index(max_value))
    return np.asarray(a)

def sentence_to_avg(sentence, word_to_vec_map):
    """
    Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
    and averages its value into a single vector encoding the meaning of the sentence.

    Arguments:
    sentence -- string, one training example from X
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation

    Returns:
    avg -- average vector encoding information about the sentence, numpy-array of shape (50,)
    """

    ### START CODE HERE ###
    # Step 1: Split sentence into list of lower case words (≈ 1 line)
    words = sentence.lower().split()

    # Initialize the average word vector, should have the same shape as your word vectors.
    avg = np.zeros(word_to_vec_map[words[0]].shape)

    # Step 2: average the word vectors. You can loop over the words in the list "words".
    for w in words:
        avg += word_to_vec_map[w]
    avg = avg/len(words)

    ### END CODE HERE ###

    return avg

def Emojify_V1(X, Y, word_to_vec_map, learning_rate = 0.01, num_iterations = 400):
    """
    Model to train word vector representations in numpy.

    Arguments:
    X -- input data, numpy array of sentences as strings, of shape (m, 1)
    Y -- labels, numpy array of integers between 0 and 7, numpy-array of shape (m, 1)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    learning_rate -- learning_rate for the stochastic gradient descent algorithm
    num_iterations -- number of iterations

    Returns:
    pred -- vector of predictions, numpy-array of shape (m, 1)
    W -- weight matrix of the softmax layer, of shape (n_y, n_h)
    b -- bias of the softmax layer, of shape (n_y,)
    """

    np.random.seed(1)

    # Define number of training examples
    m = Y.shape[0]                          # number of training examples
    n_y = 5                                 # number of classes
    n_h = 50                                # dimensions of the GloVe vectors

    # Initialize parameters using Xavier initialization
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))

    # Convert Y to Y_onehot with n_y classes
    Y_oh = convert_to_one_hot(Y, C = n_y)

    # Optimization loop
    for t in range(num_iterations):                       # Loop over the number of iterations
        for i in range(m):                                # Loop over the training examples

            ### START CODE HERE ### (≈ 4 lines of code)
            # Average the word vectors of the words from the i'th training example
            avg = sentence_to_avg(X[i], word_to_vec_map)

            # Forward propagate the avg through the softmax layer
            z = np.dot(W, avg) + b
            a = softmax(z)

            # Compute cost using the i'th training label's one hot representation and "A" (the output of the softmax)
            cost = -1*np.sum(np.multiply(Y_oh[i],np.log(a)))
            ### END CODE HERE ###

            # Compute gradients
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y,1), avg.reshape(1, n_h))
            db = dz

            # Update parameters with Stochastic Gradient Descent
            W = W - learning_rate * dW
            b = b - learning_rate * db

        if t % 100 == 0:
            print("Epoch: " + str(t) + " --- cost = " + str(cost))
            pred = predict(X, Y, W, b, word_to_vec_map)

    return pred, W, b

def predict_mode(word_to_vec_map):
    '''
    load W matrix and bias matrix from disk

    '''
    W = np.genfromtxt('data/W.csv',delimiter=',')
    b = np.genfromtxt('data/b.csv', delimiter = ',')

    # Converting the weights to tensors

    W = tf.convert_to_tensor(W, np.float64)
    b = tf.convert_to_tensor(b, np.float64)
    
    sent_avg = tf.placeholder(np.float64)
    z = tf.add(tf.tensordot(W, sent_avg, axes=1), b)
    k = tf.argmax(tf.nn.softmax(z), axis=0)

    sentence = input("Enter a sentence or type `quit` to exit \n")

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        while sentence != "quit":
            index = np.asscalar(sess.run(k, feed_dict={sent_avg: sentence_to_avg(sentence, word_to_vec_map)}))
            print(label_to_emoji(index)+"\n")
            sentence = input("Enter a sentence or type `quit` to exit \n")

if __name__ == '__main__':


    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
    predict_mode(word_to_vec_map)
    exit()
    X_train, Y_train = read_csv('data/dataTrain.csv')
    X_test, Y_test = read_csv('data/dataTest.csv')
    pred, W, b = Emojify_V1(X_train, Y_train, word_to_vec_map)
    np.savetxt("W.csv",W,delimiter=',')
    np.savetxt("b.csv",b,delimiter=',')
    pred = predict(X_test, Y_test, W, b, word_to_vec_map)
    for iter in range(len(X_test)):
        print(X_test[iter], label_to_emoji(Y_test[iter]), label_to_emoji(pred[iter]))

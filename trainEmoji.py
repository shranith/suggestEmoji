import numpy as np
import tensorflow as tf
import emoji

def load_datasets():
    f = open('data/X_train.txt','r')
    X_train = np.asarray(f.readlines())

    f = open('data/Y_train.txt','r')
    Y_train = np.asarray(f.readlines())

    f = open('data/X_test.txt','r')
    X_test = np.asarray(f.readlines())

    f = open('data/Y_test.txt','r')
    Y_test = np.asarray(f.readlines())

    return X_train, Y_train, X_test, Y_test



if __name__ == '__main__':
    print(load_datasets())

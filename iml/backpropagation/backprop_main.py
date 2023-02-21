import backprop_data

import backprop_network

import numpy as np
import matplotlib.pyplot as plt

def q2():
    '''
    Plots the training accuracy, training loss, and test accuracy as a function of epochs for different learning rates.
    '''
    training_data, test_data = backprop_data.load(train_size=10000,test_size=5000)
    X = np.arange(30)
    rates = [0.001, 0.01, 0.1, 1, 10, 100]

    final_train_acc = [None] * len(rates)
    final_train_loss = [None] * len(rates)
    final_test_acc = [None] * len(rates)
    leg = [str(rate) for rate in rates]
    for i,rate in enumerate(rates):
        net = backprop_network.Network([784, 40, 10])
        train_acc, train_loss, test_acc = net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=rate, test_data=test_data,section=2)
        final_train_acc[i] = train_acc
        final_train_loss[i] = train_loss
        final_test_acc[i] = test_acc

    for i,rate in enumerate(rates):
        plt.plot(X,final_train_acc[i])
    plt.legend(leg)
    plt.title("Training Accuracy vs Epochs")
    plt.show()
    for i,rate in enumerate(rates):
        plt.plot(X,final_train_loss[i])
    plt.legend(leg)
    plt.title("Training Loss vs Epochs")
    plt.show()
    for i,rate in enumerate(rates):
        plt.plot(X,final_test_acc[i])
    plt.legend(leg)
    plt.title("Test Accuracy vs Epochs")
    plt.show()
    
def q3():
    '''
    Prints the accuary of the last epoch.
    '''
    training_data, test_data = backprop_data.load(train_size=50000, test_size=10000)
    net = backprop_network.Network([784, 40, 10])
    net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data,section=1)

q3()

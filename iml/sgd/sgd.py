import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels



def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    #data is 6000 x 784
    N = len(data)

    w = np.zeros(784,dtype=np.float64) #w1 = 0
    for t in range(1,T+1):
        eta = eta_0 / t #update eta
        i = np.random.randint(0,N) #choose uniformly from {1,...,n}
        val = labels[i] * (np.dot(data[i],w)) # y_i * (<x_i,w>). 

        if val < 1: #update w
            w = (1-eta)*w + eta*C*labels[i]*data[i]
        else:
            w = (1-eta)*w

    return w

def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    """
    from scipy.special import softmax
    N = len(data)

    w = np.zeros(784,dtype=np.float64) #w1 = 0
    for t in range(1,T+1):
        eta = eta_0 / t #update eta
        i = np.random.randint(0,N) #choose uniformly from {1,...,n}

        x = labels[i] - labels[i]*sigmoid(np.dot(-1*w,data[i])) # Calc gradient
        x = np.dot(data[i],x)

        w = w + eta*x #update w
        
    return w

#################################

def sigmoid(X):
    return 1/(1+np.exp(X))

import matplotlib.pyplot as plt 

def Q1a():
    '''
    Plots the average accuracy of the validation set as a function of eta for 10 runs of SGD for hinge loss.
    '''
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    etas = [10**i for i in range(-5,6)]
    avgAccs = []
    for eta in etas:
        accs = []
        for i in range(10):
            w = SGD_hinge(train_data,train_labels,1,eta,1000)
            
            # Calculate acc using 0-1 loss:
            counter = 0
            M = len(validation_data)
            for j in range(M):
                val = np.inner(validation_data[j],w)
                if val >= 0:
                    sign = 1
                else:
                    sign = -1
                
                if sign == validation_labels[j]:
                    counter += 1
            accs.append(counter/M)
        avgAccs.append(np.average(accs))
    
    plt.plot(etas,avgAccs)
    plt.xscale('log')
    plt.show()

def Q1b():
    '''
    Plots the average accuracy of the validation set as a function of C for 10 runs of SGD for hinge loss.
    '''
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    Cs = [10**i for i in range(-5,6)]
    avgAccs = []
    for C in Cs:
        accs = []
        for i in range(10):
            w = SGD_hinge(train_data,train_labels,C,1,1000)
            
            # Calculate acc using 0-1 loss:
            counter = 0
            M = len(validation_data)
            for j in range(M):
                val = np.inner(validation_data[j],w)
                if val >= 0:
                    sign = 1
                else:
                    sign = -1
                
                if sign == validation_labels[j]:
                    counter += 1
            accs.append(counter/M)
        avgAccs.append(np.average(accs))
    
    plt.plot(Cs,avgAccs)
    plt.xscale('log')
    plt.show()

def Q1c():
    '''
    Plots a 28x28 image of the weight vector w found by SGD.
    '''
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    w = SGD_hinge(train_data,train_labels,(1/10**4),1,20000)
    plt.imshow(np.reshape(w,(28,28)),interpolation='nearest')
    plt.show()

def Q1d():
    '''
    Finds the accuracy of the best classifier on the test set.
    '''
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    w = SGD_hinge(train_data,train_labels,(1/10**4),1,20000)
    counter = 0
    M = len(test_data)
    for j in range(M):
        val = np.inner(test_data[j],w)
        if val >= 0:
            sign = 1
        else:
            sign = -1
        
        if sign == test_labels[j]:
            counter += 1
    print(counter/M)

def Q2a():
    '''
    Plots the average accuracy of the validation set as a function of eta for 10 runs of SGD for log loss.
    '''
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    etas = [10**i for i in range(-5,6)]
    avgAccs = []
    for eta in etas:
        accs = []
        for i in range(10):
            w = SGD_log(train_data,train_labels,eta,1000)
            
            # Calculate acc using 0-1 loss:
            counter = 0
            M = len(validation_data)
            for j in range(M):
                val = np.inner(validation_data[j],w)
                if val >= 0:
                    sign = 1
                else:
                    sign = -1
                
                if sign == validation_labels[j]:
                    counter += 1
            accs.append(counter/M)
        avgAccs.append(np.average(accs))
    
    plt.plot(etas,avgAccs)
    plt.xscale('log')
    plt.show()

def Q2b():
    '''
    Plots a 28x28 image of the weight vector w found by SGD.
    '''
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    w = SGD_log(train_data,train_labels,1/(10**4),20000)
    plt.imshow(np.reshape(w,(28,28)),interpolation='nearest')
    plt.show()
    counter = 0
    M = len(test_data)
    for j in range(M):
        val = np.inner(test_data[j],w)
        if val >= 0:
            sign = 1
        else:
            sign = -1
        
        if sign == test_labels[j]:
            counter += 1
    print(counter/M)


def Q2c():
    '''
    Plots the norm of the weight vector as a function of the number of iterations.
    '''
    from numpy import linalg as LA
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    iterations = [i for i in range(1000,20001,1000)]
    norms = []
    for i in iterations:
        w = SGD_log(train_data,train_labels,10**(-4),i)
        norms.append(LA.norm(w))

    plt.plot(iterations,norms)
    plt.show()

#################################

Q2c()

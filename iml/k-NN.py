from ast import operator
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']

import numpy as np
idx = np.random.RandomState(0).choice(70000,11000)
train = data[idx[:1000],:].astype(int)
train_labels = labels[idx[:1000]]
test = data[idx[1000:],:].astype(int)
test_labels = labels[idx[1000:]]

def KNNPrediction(tImages,vLabels,queryImage,k):
    N = len(tImages)
    distances = [] # [dist,label]

    for i in range(N): 
        dist = np.linalg.norm(tImages[i]- queryImage)
        distances.append([dist,vLabels[i]])
    
    distances.sort(key = lambda x: x[0]) #sort by distance
    distances = distances[:k] #only want the closest k

    pLabels = [0 for i in range(10)] #10 possible labels

    for d,l in distances: #count per label
        pLabels[int(l)] += 1
    
    return np.argmax(pLabels) #find the best label

### B
'''
finds the error rate of the KNN prediction with k = 10 and 1000 training images
'''
errors = 0
k = 10
for i in range(1000):
    if KNNPrediction(train,train_labels,test[i],k) != int(test_labels[i]):
        errors += 1

print(errors) 

import matplotlib.pyplot as plt

### C
'''
plots the error rate of the KNN prediction with k = 1,2,...,100 and 1000 training images
'''

kMap = {} #maps k value -> error %
for k in range(1,101):
    errors = 0
    for i in range(1000):
        if KNNPrediction(train,train_labels,test[i],k) != int(test_labels[i]):
            errors += 1
    kMap[k] = (errors/10)
a,b = zip(*list(kMap.items()))
plt.plot(a,b)
plt.xlabel("k value")
plt.ylabel("Error precentage")

plt.show() 

### D
'''
plots the error rate of the KNN prediction with number of training images = 100,200,...,5000 and k = 1
'''
k = 1
nMap = {} #maps n value -> error %
for n in range(100,5001,100):
    train = data[idx[:n],:].astype(int)
    train_labels = labels[idx[:n]]
    test = data[idx[n:],:].astype(int)
    test_labels = labels[idx[n:]]
    errors = 0
    for i in range(n):
        if KNNPrediction(train,train_labels,test[i],k) != int(test_labels[i]):
            errors += 1
    nMap[n] = ((errors*100)/n)
a,b = zip(*list(nMap.items()))
plt.plot(a,b)
plt.xlabel("n value")
plt.ylabel("Error precentage")

plt.show()



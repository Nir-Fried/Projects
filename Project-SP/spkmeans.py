import sys
import math
import numpy as np
import pandas as pd
import spkmeans as spk

if len(sys.argv) != 4:
    print('Invalid input!')
    quit()

K = sys.argv[1]
goal = sys.argv[2]
file_name = sys.argv[3]

if K.isdigit()==False: #if K isn't an int
    print('Invalid input!')
    quit()

if goal not in ['spk','wam','ddg','lnorm','jacobi']:
    print('Invalid input!')
    quit()

K = int(K)

print("-----------startPYTHON-----------")

d=1 #calculate d
try:
    file = open(file_name,'r')
    line = file.readline()
    for chars in line:
        if chars==',':
            d=d+1
        file.close()
except FileNotFoundError:
    print('An Error Has Occurred')
    quit()

N = 0 #calculate d
file = open(file_name,'r')
line = file.readline()
while line !='':
    line=file.readline()
    N=N+1
file.close()

if K<0 or K>=N:
    print('Invalid Input!')
    quit() 


header = [] #transform given file to a pd table
header.append("0")
for i in range(d-1):
    header.append(str(i+1))
input = pd.read_csv(file_name,sep=',',names=header)

print(input)


index = 0
dataPoints = [0 for y in range(N*d)]
observations = [-1]
for x in input.values.tolist():
    for y in x:
        dataPoints[index] = y
        index = index + 1

if goal in ['wam','ddg','lnorm','jacobi']: #if we dont need to determine K or to run Kmeans
    spk.wam(K,d,N,observations,dataPoints,goal)
    quit()

if K==0: #if we need to determine K
    K = spk.wam(K,d,N,observations,dataPoints,goal)

spk.wam(K,d,N,observations,dataPoints,goal)

print("-----------startKmeans++-----------")

NN = 0
file2 = open("nirTestFile.txt",'r')
line = file2.readline()
while line !='':
    line = file2.readline()
    NN = NN + 1
file2.close()

dd=0
file2 = open("nirTestFile.txt",'r')
line = file2.readline()
for chars in line:
    if chars==',':
        dd = dd +1
    file.close()

print("dd is " , dd , " and NN is ", NN)


header = [] #transform given file to a pd table
header.append("0")
for i in range(dd-1):
    header.append(str(i+1))
input2 = pd.read_csv("nirTestFile.txt",sep=',',names=header,index_col=None)
print("yesssssss\n")
print(input2)

observations.remove(-1) #remove the -1, now its empty.

#structures:
centroids = [0 for y in range(K)]
prob = [0 for y in range(NN)]
probabilites = [0 for y in range(NN)]
D = [0 for y in range(NN)]


#K-means++ 
i=1
np.random.seed(0)
rand = np.random.randint(low=0,high=N)
centroids[0] = input2.iloc[rand,1:dd+1].values
observations.append(rand)

while i<K: #repeat
    for l in range(N):
        min = ((input2.iloc[l,1:dd+1].values-centroids[0])**2)
        for j in range(i-1):
            val = ((input2.iloc[l,1:dd+1].values-centroids[j+1])**2)
            if sum(val[0])<sum(min):
                min = val[0]
        D[l] = min
    sum1 = 0
    for m in range(N):
        sum1 = sum1 + sum(D[m])

    for l in range(N):
        prob[l] = D[l]/sum1
    
    for l in range(N):
        probabilites[l] = sum(prob[l]) #sum the probabilities

    a = np.random.choice(N,1,replace=False,p = probabilites) #random observation
    centroids[i] = input2.iloc[a,1:dd+1].values
    observations.append(a[0])
    i = i +1

print(','.join(map(str,observations)))

print("-----------endKmeans++-----------")

spk.wam(K,d,N,observations,dataPoints,goal)

print("-----------endPYTHON-----------")
quit()
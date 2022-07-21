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
observations = []
for x in input.values.tolist():
    for y in x:
        dataPoints[index] = y
        index = index + 1

if goal in ['wam','ddg','lnorm','jacobi']: #if we dont need to determine K or run Kmeans
    spk.wam(K,d,N,observations,dataPoints,goal)
    quit()


if K==0: #if we need to determine K
    K = spk.wam(K,d,N,observations,dataPoints,goal)

#structures:
centroids = [0 for y in range(K)]
prob = [0 for y in range(N)]
probabilites = [0 for y in range(N)]
D = [0 for y in range(N)]

#K-means++ 
i=1
np.random.seed(0)
rand = np.random.randint(low=0,high=N)
centroids[0] = input.iloc[rand,0:d].values
observations.append(rand)

while i<K: #repeat
    for l in range(N):
        min = ((input.iloc[l,0:d].values-centroids[0])**2)
        for j in range(i-1):
            val = ((input.iloc[l,0:d].values-centroids[j+1])**2)
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
    centroids[i] = input.iloc[a,0:d].values
    observations.append(a[0])
    i = i +1


print(','.join(map(str,observations)))
spk.wam(K,d,N,observations,dataPoints,goal)

print("-----------endPYTHON-----------")
import sys
import math
import numpy as np
import pandas as pd
import mykmeanssp as kmeanspp

def isFloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

if len(sys.argv) != 5 and len(sys.argv) != 6:
    print('Invalid input!')
    quit()

if len(sys.argv) == 5: #without max_iter
    K = sys.argv[1]
    max_iter = "300" #if not provided, max_iter = 300
    epsilon = sys.argv[2]
    input1 = sys.argv[3]
    input2 = sys.argv[4]

elif len(sys.argv) == 6:
    K = sys.argv[1]
    max_iter = sys.argv[2]
    epsilon = sys.argv[3]
    input1 = sys.argv[4]
    input2 = sys.argv[5]

#Calculate d and N.

d=0
try:
    file = open(input1,'r')
    line = file.readline()
    for chars in line:
        if chars==',':
            d=d+1
        file.close()
except FileNotFoundError:
    print('An Error Has Occurred')
    quit()

N = 0
try:
    file = open(input2,'r')
    line = file.readline()
    while line !='':
        line = file.readline()
        N = N +1
    file.close()
except FileNotFoundError:
    print('An Error Has Occurred')
    quit()

#check validation of inputs.

if K.isdigit()==False: #checking if K in an integer
    print('Invalid input!')
    quit()

if max_iter.isdigit()==False: #checking if max_iter is an integer
    print('Invalid Input!')
    quit()

if isFloat(epsilon)==False:
    print('Invalid Input!')
    quit()

#   Converting K and max_iter to ints:
K = int(K)
max_iter = int(max_iter)

#convert epsilon to float:
epsilon = float(epsilon)

if K<2 or K>=N or max_iter<1:
    print('Invalid Input!')
    quit()


#merging the given files
header = []
header.append("key")
for i in range(d):
    header.append(str(i))

table1 = pd.read_csv(input1,sep=',',names=header)
table2 = pd.read_csv(input2,sep=',',names=header)


merged = pd.merge(table1,table2,on='key')
#print(merged)

d = 2*d #after the merge d is doubled.

merged = merged.sort_values('key') #sorting by key value to make things easier later
#print(merged)

#structures:
centroids = [0 for y in range(K)]
prob = [0 for y in range(N)]
probabilites = [0 for y in range(N)]
D = [0 for y in range(N)]
observations = []
dataPoints = [0 for y in range(N*d)]

#K-means++ 
i=1
np.random.seed(0)
rand = np.random.randint(low=0,high=N)
centroids[0] = merged.iloc[rand,1:d+1].values
observations.append(rand)

while i<K: #repeat
    for l in range(N):
        min = ((merged.iloc[l,1:d+1].values-centroids[0])**2)
        for j in range(i-1):
            val = ((merged.iloc[l,1:d+1].values-centroids[j+1])**2)
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
    centroids[i] = merged.iloc[a,1:d+1].values
    observations.append(a[0])
    i = i +1

print(','.join(map(str,observations)))

index = 0
for x in merged.iloc[0:N,1:d+1].values.tolist():
    for y in x:
        dataPoints[index] = y
        index = index +1

kmeanspp.fit(K,d,N,max_iter,epsilon,observations,dataPoints)

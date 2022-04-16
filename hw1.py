import sys
import math

print('-------------- start ----------------')

if len(sys.argv) != 4 and len(sys.argv) != 5:
    print('Invalid input!')
    quit()

if len(sys.argv) == 4: #without max_iter
    K = sys.argv[1]
    max_iter = "200" #if not provided, max_iter = 200
    input = sys.argv[2]
    output = sys.argv[3]
elif len(sys.argv)==5: #with max_iter
    K = sys.argv[1]
    max_iter = sys.argv[2]
    input = sys.argv[3]
    output = sys.argv[4]

print('K=' , K , ' max_iter=' , max_iter , ' input=' , input , ' output=', output)
print('1-----------------------1')

#Calculate d and N.

d=1
try:
    file = open(input,'r')
    line = file.readline()
    for chars in line:
        if chars==',':
            d = d+1
    file.close()
except FileNotFoundError:
    print('An Error Has Occurred')
    quit()

N=0
file = open(input,'r')
line = file.readline()
while line !='':
    #print(line)
    line=file.readline()
    N=N+1
file.close()

print('N=' , N , ' d=' , d)

#check validation of inputs.

if K.isdigit()==False: #checking if K in an integer
    print('Invalid input!')
    quit()

if max_iter.isdigit()==False: #checking if max_iter is an integer
    print('Invalid Input!')
    quit()

#   Converting K and max_iter to ints:
K = int(K)
max_iter = int(max_iter)

if K<2 or K>=N or max_iter<0:
    print('Invalid Input!')
    quit()

print('2-----------------------2')

dataPoints = [[0 for x in range(d)] for y in range(N)]
centroids = [[0 for x in range(d)] for y in range(K)]
centroidsHolder = [[0 for x in range(d)] for y in range(K)]
numInCluster = [0 for x in range(K)]
clusters = [[[0 for x in range(d)] for y in range(N)] for z in range(K)]

file = open(input,'r') #init dataPoints
line = file.readline()
x=0
while line !='':
    lineList = line.split(",")
    lineList[len(lineList)-1] = lineList[len(lineList)-1].rstrip("\n")
    for y in range(d):
        #print('x= ', x , ' y = ', y , ' lineList = ', lineList)
        dataPoints[x][y] = float(lineList[y])
    x = x+1
    line=file.readline()
    
file.close()

for x in range(N): # printing dataPoints: delete this block later ~~~~~~~~
    for y in range(d):
        print(dataPoints[x][y], end=" ")
    print("")


print('3-----------------------3')

for x in range(K): #init centroids
    for y in range(d):
        centroids[x][y] = dataPoints[x][y]
       
for x in range(K): # printing centroids: delete this block later ~~~~~~~~
    for y in range(d):
        print(centroids[x][y], end=" ")
    print("")

print('4-----------------------4')

flag = True
iter = 0

while flag==True and iter<max_iter: #repeat:

    for x in range(K):
        for y in range(N):
            for z in range(d):
                clusters[x][y][z] = 0

    for x in range(K):
        numInCluster[x] = 0


    for x in range(N):
        min = 10000000000
        for y in range(K):
            sum = 0
            for z in range(d):
                #print('data points: ', dataPoints[y][z] , ' centoids: ' , centroids[x][z])
                sum = sum + ((dataPoints[x][z]-centroids[y][z])**2)
            #print(sum)    
            if sum<min:
                min = sum
                minIndex = y
                #print('minIndex = ', minIndex , 'min= ' , min)
        
        for z in range(d):
            temp = numInCluster[minIndex]
            clusters[minIndex][x][z] = dataPoints[x][z]
            numInCluster[minIndex] = numInCluster[minIndex]+1

    #update the centroids
    # first we copy the current centroids to another array in order to check if new-old<epsilon) 
    for x in range(K):
        for y in range(d):
            centroidsHolder[x][y] = centroids[x][y]

    #calculate new centroids
    for x in range(K):
        count = 0
        while count<d:
            sum = 0
            for y in range(N):
                sum = sum + clusters[x][y][count]
            centroids[x][count] = sum/(numInCluster[x]/d)
            #print(centroids[x][count])
            count = count+1
    
    # now we want to check if ||new-old||<epsilon for every vector in centroids[][]
    for x in range(K):
        for y in range(d):
            centroidsHolder[x][y] = abs(centroidsHolder[x][y]) - abs(centroids[x][y])
    
    counter1=0
    for x in range(K):
        sum =0
        for y in range(d):
            sum = sum + (centroidsHolder[x][y]**2)
        norm = math.sqrt(sum)
        if norm < 0.001:
            counter1 = counter1 + 1
            if counter1==K:
                flag = False
    

    iter = iter+1

for x in range(K):
        print(centroids[x])

print('5-----------------------5')

file = open(output,'w')
for x in range(K):
    line = ""
    for y in range(d):
        centroids[x][y] = round(centroids[x][y],4)
        line = line + str(centroids[x][y])
        if y != d-1:
            line = line + ","
    line = line + "\n"
    file.write(line)
file.close()


print('-------------- final ----------------')

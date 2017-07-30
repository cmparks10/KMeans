import numpy as np
from pprint import pprint
import random
import sys
import warnings

arglist = sys.argv 

#UNCOMMENT BELOW IN FINAL PROGRAM
'''
NoOfCentroids = int(arglist[2])
dataPointsFromFile = np.array(np.loadtxt(sys.argv[1], delimiter = ','))
'''

dataPointsFromFile = np.array(np.loadtxt('iris.txt', delimiter = ','))

NoOfCentroids = input('How Many Centrouds? ')
                            
dataRange = ([])

#UNCOMMENT BELOW IN FINAL PROGRAM
'''
with open(arglist[1]) as f:
    print 'Points in data set: ',sum(1 for _ in f)
'''
dataRange.append(round(np.amin(dataPointsFromFile),1))
dataRange.append(round(np.amax(dataPointsFromFile),1))
dataRange = np.asarray(dataRange)

dataPoints = np.array(dataPointsFromFile)
print 'Dimensionality of Data: ', dataPoints.shape

randomCentroids = []

templist = []
i = 0

while i<NoOfCentroids:
    for j in range(len(dataPointsFromFile[1,:])):
        cat = round(random.uniform(np.amin(dataPointsFromFile),np.amax(dataPointsFromFile)),1)
        templist.append(cat)
    randomCentroids.append(templist)
    templist = []
    i = i+1

centroids = np.asarray(randomCentroids)

def kMeans(array1, array2):
    ConvergenceCounter = 1
    keepGoing = True
    StartingCentroids = np.copy(centroids)
    print 'Starting Centroiuds:\n {}'.format(StartingCentroids)
    while keepGoing:      
        #--------------Find The new means---------#
        t0 = StartingCentroids[None, :, :] - dataPoints[:, None, :]
        t1 = np.linalg.norm(t0, axis=-1)
        t2 = np.argmin(t1, axis=-1)
        #------Push the new means to a new array for comparison---------#
        CentroidMeans = []
        for x in range(len(StartingCentroids)):
            if np.any(t2==[x]):
                CentroidMeans.append(np.mean(dataPoints[t2 == [x]], axis=0))
        #--------Convert to a numpy array--------#
        NewMeans = np.asarray(CentroidMeans)
        #------Compare the New Means with the Starting Means------#
        if np.array_equal(NewMeans,StartingCentroids):
            print ('Convergence has been reached after {} moves'.format(ConvergenceCounter))
            print ('Starting Centroids:\n{}'.format(centroids))
            print ('Final Means:\n{}'.format(NewMeans))
            print ('Final Cluster assignments: {}'.format(t2))
            for x in xrange(len(StartingCentroids)):
                print ('Cluster {}:\n'.format(x)), dataPoints[t2 == [x]]
            for x in xrange(len(StartingCentroids)):
                print ('Size of Cluster {}:'.format(x)), len(dataPoints[t2 == [x]])
            keepGoing = False
        else:
            #print 15*'-'
            ConvergenceCounter  = ConvergenceCounter +1
            #print 'Starting Centroids:\n'
            #print StartingCentroids
            #print '\n'
            #print 'Starting NewMeans:\n'
            #print NewMeans
            StartingCentroids =np.copy(NewMeans)
            #print 'Starting Centroids Now:\n'
            #print StartingCentroids
            #print '\n'
            #print 'NewMeans now:'
            #print NewMeans
            #break


kMeans(centroids, dataPoints)

        
'''
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    kMeans(centroids,dataPoints) 


def IrisKMeans(Dataset): #Pass the dataset only to the function, clusters will be made inside
    NoOfCentroids = 3
    #----Create centroid points----#
    rangeX = (0, 8)
    rangeY = (0, 8)
    rangeA = (0, 8)
    rangeB = (0, 8)
    
    dataPoints = np.copy(irisData)
    randomCentroids = []
    i = 0
    while i<NoOfCentroids:
        x = random.randrange(*rangeX)
        y = random.randrange(*rangeY)
        a = random.randrange(*rangeA)
        b = random.randrange(*rangeB)
        randomCentroids.append((x,y,a,b))
        i += 1
    centroids = np.asarray(randomCentroids)
    print 'Random Centroids Chosen:\n',centroids
    
    ConvergenceCounter = 1
    keepGoing = True
    StartingCentroids = np.copy(centroids)
    while keepGoing:      
        #--------------Find The new means---------#
        #np.linalg.norm(StartingCentroids[None, :, :] - dataPoints[:, None, :], axis=-1)
        t0 = StartingCentroids[None, :, :] - dataPoints[:, None, :]
        t1 = np.linalg.norm(t0, axis=-1)
        t2 = np.argmin(t1, axis=-1)
        #------Push the new means to a new array for comparison---------#
        CentroidMeans = []
        for x in xrange(len(StartingCentroids)):
            CentroidMeans.append(np.mean(dataPoints[t2 == [x]], axis=0))
        #--------Convert to a numpy array--------#
        NewMeans = np.asarray(CentroidMeans)
        #------Compare the New Means with the Starting Means------#
        if np.array_equal(NewMeans,StartingCentroids):
            #print ('Convergence has been reached after {} moves'.format(ConvergenceCounter))
            #print ('Starting Centroids:\n{}'.format(centroids))
            print ('Final Means:\n{}'.format(NewMeans))
            maxValues = map(max,zip(*NewMeans))
            print 'Maximum Overlap:\n',maxValues
            print 'Purity Score:',sum(maxValues)/150
            #print ('Final Cluster assignments: {}'.format(t2))
            #for x in xrange(len(StartingCentroids)):
            #    print ('Cluster {}:\n'.format(x)), dataPoints[t2 == [x]]
            #for x in xrange(len(StartingCentroids)):
            #    print ('Size of Cluster {}:'.format(x)), len(dataPoints[t2 == [x]])
            keepGoing = False
        else:
            ConvergenceCounter  = ConvergenceCounter +1
            StartingCentroids =np.copy(NewMeans)
print ('\n')
with open('iris.txt') as iris:
    print 'Points in iris data set: ',sum(1 for _ in iris)

IrisKMeans(irisData)
'''

########################################################
#File: train.py
#Description : Trains the neural network with a 2-5-4
#configuration with 0,10,100,1000 and 10000 epoch values.
#
#Author : Piyush Verma
#
#
#
#########################################################




import random
import math
import pickle
from matplotlib import pyplot

#global variables
trainingList = []
weightMatrix = {}
plotGraph = {}


#initializes the weight matrix with random values between -1 and 1
def initializeWeightMatrix(hiddenMin, hiddenMax, outputMin, outputMax):
    global weightMatrix

    temp1 = {}
    temp2 = {}
    for inputNode in range(0,3):
        for hiddenNode in range(hiddenMin, hiddenMax+1):
            temp2[hiddenNode] = random.uniform(-1,1)
        temp1[inputNode] = temp2
        temp2 = {}            
    
    for hiddenNode in range(hiddenMin, hiddenMax+2):
        for outputNode in range(outputMin, outputMax+1):
            temp2[outputNode] = random.uniform(-1,1)
        temp1[hiddenNode] = temp2
        temp2 = {}
        
    weightMatrix = temp1


#reads the training data from the csv file
def readTrainingData():
    global trainingList
    
    
    inputFile = open('train_data.csv')
    
    for line in inputFile:
        
        tempList = []
        
        linePointer = 0
        for i in range(0,2):
            commaLocation = line.find(",",linePointer,len(line))
            temp = line[linePointer:commaLocation]
            linePointer = commaLocation + 1
            tempList.append(float(temp))
            
        temp = line[commaLocation + 1:(len(line)-1)]
        tempList.append(temp)
        #print tempList
        trainingList.append(tempList)
        
    

#calculates the hidden node values

def calculateHiddenNode(outputNode, index, inputBias):

    global trainingList
    sumOfWeights = 0    
    for i in range(0, 2):
        sumOfWeights = sumOfWeights + (trainingList[index][i]*weightMatrix[i][outputNode])
    sumOfWeights = sumOfWeights + weightMatrix[inputBias][outputNode]
    sigmoid = float(1/(1+math.exp((-1)*sumOfWeights)))
    return sigmoid

#calculate the output node values
def calculateOutputNode(outputNode, hiddenBias, hiddenMin, hiddenMax, hiddenValues):

    global trianingList
    sumOfWeights = 0    
    for hiddenNode in range(hiddenMin, hiddenMax+1):
        sumOfWeights = sumOfWeights + (hiddenValues[hiddenNode]*weightMatrix[hiddenNode][outputNode])
        
    sumOfWeights = sumOfWeights + weightMatrix[hiddenBias][outputNode] 
    
    sigmoid = float(1/(1+math.exp((-1)*sumOfWeights)))
    return sigmoid

def calculateSSESum(targetOutput, outputValues):
    sseSum = 0
    targetIndex = 0
    for node in sorted(outputValues):
        sseSum = sseSum + (outputValues[node] - targetOutput[targetIndex])**2
        targetIndex = targetIndex + 1
    
    return float(sseSum)/2
    

    
#calculates the values at nodes, delta errors and updates the weights for each node in the neural network.
def backPropagate(hiddenMin, hiddenMax, outputMin, outputMax, inputBias, hiddenBias, outFileName, epoch):
    plotGraph['x'] = []
    plotGraph['y'] = []
    sseSum = 0
    
    #Repeat for the given epoch value
    for epochCount in range(0,epoch):
        
        
        
        sseSum = 0
        
        
        
        #Repeat over all training trainData
        for trainData in range(0,74):
            
            
            #initializing Delta List
            delta = {}
            for i in range(3, outputMax+1):
                if(i == hiddenBias):
                    pass
                else:
                    delta[i] = 0
    
            #Calculating output at the hidden layer
            hiddenValues = {}
            for node in range(hiddenMin, hiddenMax+1):
                hidden = calculateHiddenNode(node, trainData, inputBias)
                hiddenValues[node] = hidden
                
            #Calculating output at the output layer
            outputValues = {}
            for node in range(outputMin, outputMax+1):#outputNodes:
                hidden = calculateOutputNode(node, hiddenBias, hiddenMin, hiddenMax, hiddenValues)
                outputValues[node] = hidden
            
            
            
            targetOutput = []
            realOutput = trainingList[trainData][2]
            
            if(realOutput == str(1)):
                targetOutput = [1,0,0,0]        
            elif(realOutput == str(2)):
                targetOutput = [0,1,0,0]
            elif(realOutput == str(3)):
                targetOutput = [0,0,1,0]
            elif(realOutput == str(4)):
                targetOutput = [0,0,0,1]
            
            #Calculating delta at the output node
            targetIndex = 0            
            for node in range(outputMin, outputMax+1):
                delta[node] = (outputValues[node] - targetOutput[targetIndex])*outputValues[node]*(1-outputValues[node])
                targetIndex = targetIndex + 1
            
            #Calculating deltas at the hidden nodes
            tempDeltaSum = 0
            for hidden in range(hiddenMin, hiddenMax+1):#node 3 and node 4
                for output in range(outputMin, outputMax+1):
                    tempDeltaSum = tempDeltaSum + delta[output]*weightMatrix[hidden][output]
                delta[hidden] = tempDeltaSum*hiddenValues[hidden]*(1-hiddenValues[hidden])
                tempDeltaSum = 0
            
            #learning rate
            alpha = 0.1
            
            #hidden bias
            hiddenValues[hiddenBias] = 1 
            
            #Updating weights between hidden and output nodes
            for hidden in range(hiddenMin, hiddenMax+2):
                for output in range(outputMin, outputMax+1):
                    weightMatrix[hidden][output] = weightMatrix[hidden][output] - (alpha*(delta[output])*(hiddenValues[hidden]))
                    
            #updating weights between hidden and input layers
            inputIndex = 0            
            for input in range(0,2):
                for hidden in range(hiddenMin, hiddenMax+1):
                    weightMatrix[input][hidden] = weightMatrix[input][hidden] - (alpha*delta[hidden]*trainingList[trainData][inputIndex])
                inputIndex = inputIndex+1
            
            for hidden in range(hiddenMin, hiddenMax+1):
                weightMatrix[2][hidden] = weightMatrix[2][hidden] - (alpha*delta[hidden]) 
                
            tempSSE = calculateSSESum(targetOutput, outputValues)
            sseSum = sseSum + tempSSE
        
        plotGraph['x'].append(epochCount)
        plotGraph['y'].append(sseSum)
        
    outputFile = open(outFileName, 'w')
    print "Creating file: "+ outFileName
    pickle.dump(weightMatrix, outputFile)
    print "File created"
    outputFile.close()
    if(epoch == 1000 or epoch == 10000):
        pyplot.plot(plotGraph['x'], plotGraph['y'])
        pyplot.xlabel("Epoch Count")
        pyplot.ylabel("Sum of Squared Errors")
        pyplot.savefig("graph_epoch"+str(epoch)+".png")
        pyplot.show()
    
def main():

    global weightMatrix
    global plotGraph

    readTrainingData()
    print "Training the Neural Network with 2-5-4 configuration."
    
    initializeWeightMatrix(3, 7, 9, 12)
    print "Generating file for 0 epochs..."
    backPropagate(3, 7, 9, 12, 2, 8, 'weightMatrix1.txt', 0)
    print "File generated successfully!"
    
    initializeWeightMatrix(3, 7, 9, 12)
    print "Generating file for 10 epochs..."
    backPropagate(3, 7, 9, 12, 2, 8, 'weightMatrix2.txt', 10)
    print "File generated successfully!"
    
    initializeWeightMatrix(3, 7, 9, 12)
    print "Generating file for 100 epochs..."
    backPropagate(3, 7, 9, 12, 2, 8, 'weightMatrix3.txt', 100)
    print "File generated successfully!"
    
    initializeWeightMatrix(3, 7, 9, 12)
    print "Generating file for 1000 epochs..."
    backPropagate(3, 7, 9, 12, 2, 8, 'weightMatrix4.txt', 1000)
    print "File generated successfully!"
    
    initializeWeightMatrix(3, 7, 9, 12)
    print "Generating file for 10000 epochs..."
    backPropagate(3, 7, 9, 12, 2, 8, 'weightMatrix5.txt', 10000)
    print "File generated successfully!"
    

if(__name__ == "__main__"):
    main()

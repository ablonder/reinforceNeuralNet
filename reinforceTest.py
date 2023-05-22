# reinforceTest.py
# Aviva Blonder
# Several functions that test the neuralnet and pool classes in reinforcenet.py

import numpy
import random
import reinforcenet

# A simple binary problem
def binary():
    nn = reinforcenet.neuralnet(.05, [2, 2])
    for i in range(0, 10):
        print()
        act = nn.feedforward([1, 0])
        print(act)
        if act == 0:
            nn.backpropagate(1)
        else:
            nn.backpropagate(-1)
        act = nn.feedforward([0, 1])
        print(act)
        if act == 1:
            nn.backpropagate(1)
        else:
            nn.backpropagate(-1)
##        act = nn.feedforward([1, 1])
##        print(act)
##        if act == 0:
##            nn.backpropagate(1)
##        else:
##            nn.backpropagate(0)
##        act = nn.feedforward([0, 0])
##        print(act)
##        if act == 0:
##            nn.backpropagate(1)
##        else:
##            nn.backpropagate(0)


# The monks problem
def monks():
    # preprocessing taken from homework 3
    file = open("monks1.csv")
    attrvals = {}
    data = []
    start = True
    for line in file:
        line = line.strip()
        line = line.split(",")
        # process the first line as a list of attributes
        if start:
            attributes = line
            # prepare attrvals to get a list of values for each attribute
            for attr in attributes:
                attrvals[attr] = []
            start = False
        # interpret all other lines as instances
        else:
            # add each instance to data
            data.append(line)
            # go through each attribute value and add it to attrvals if it
            ## isn't there already
            for i in range(len(attributes)):
                if line[i] not in attrvals[attributes[i]]:
                    attrvals[attributes[i]].append(line[i])

    # shuffle data into train and test sets
    random.seed(1)
    random.shuffle(data)
    split = .8
    trainsize = int(split*len(data))
    trainset = data[:trainsize]
    testset = data[trainsize:]

    # preprocess the train and test sets into binary vectors of attributes and labels
    traininst, trainlabels = preprocess(trainset, attributes, attrvals)
    testinst, testlabels = preprocess(testset, attributes, attrvals)

    # create a neural network and test it
    nn = reinforcenet.neuralnet(.1, [len(traininst[0]), 10, 10, 1])
    # lrate = .5, avg yes error = .518, avg no error = -.515
    # lrate = .05, avg yes error = .501, avg no error = .498
    # training!
    for epoch in range(100):
        y = 0
        numy = 0
        n = 0
        numn = 0
        for inst in range(len(traininst)):
            predicted = nn.feedforward(traininst[inst])
            nn.backpropagate([trainlabels[inst]])
            #print(str(trainlabels[inst]) + " " + str(predicted))
            if trainlabels[inst] == 1:
                y += predicted[0]
                numy += 1
            else:
                n += predicted[0]
                numn += 1
        # print average error for each label
        print("yes: " + str(y/numy))
        print("no: " + str(n/numn))
        
    # testing!
##    y = 0
##    numy = 0
##    n = 0
##    numn = 0
##    for i in range(len(testinst)):
##        # turn the index provided into a label for the actual and prediction
##        reall = testlabels[i]
##        predl = nn.feedforward(testinst[i])
##        # I'm just going to be lazy and print them both
##        print(str(reall) + " " + str(predl))
##        if reall == 1:
##            y += predl[0]
##            numy += 1
##        else:
##            n += predl[0]
##            numn += 1
##    # let's also print average prediction for each label
##    print("yes: " + str(y/numy))
##    print("no: " + str(n/numn))
        

# preprocessing nominal values into continuous ones thorugh one hot coding
## taken from homework 3
def preprocess(data, attributes, attrvals):
    # loop through data to turn all attribute values and labels into vectors and
    ## remove the label from each training instance and to it to instlabels
    labels = []
    newdata = []
    for inst in data:
        newinst = []
        # add the actual label to labels
        # if the label is no, add a 0
        if inst[0] == '"no"':
            labels.append(0)
        # otherwise add a 1
        else:
            labels.append(1)
        # one hot code the rest of the attributes
        for i in range(1, len(attributes)):
            posvals = attrvals[attributes[i]]
            val = inst[i]
            # first try to turn val into a float
            try:
                vec = [float(val)]
            # if it isn't continuous, turn it into a vector using one hot coding
            except ValueError:
                vec = numpy.zeros(len(posvals))
                vec[posvals.index(val)] = 1
            # if this is the label, add it to labels
            if i == 0:
                labels.append(vec)
            # otherwise add it on to the end of the new instance
            else:
                newinst.extend(vec)
        # add the instance to new data
        newdata.append(newinst)
    # return the new data set and labels
    return newdata, labels

monks()

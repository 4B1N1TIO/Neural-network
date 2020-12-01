####################################
# Easy Neural-Network for OCR numbers
# Author:       Adrian Thierbach
# Date:         26.11.2020
# Dependencies: Python3,numpy,scipy
####################################

#Inspired by :
#-----------------------------------
# @book{rashid2017neuronale,
#  title={Neuronale Netze selbst programmieren: ein verst{\"a}ndlicher Einstieg mit Python},
#  author={Rashid, Tariq},
#  year={2017},
#  publisher={O'Reilly}
# }
#----------------------------------


import numpy
import scipy.special

class neuralnetwork:
    # initialize a 3 layered neural network
    def __init__(self,inputnodes,hiddennodes,outputnodes,mixingfactor):
        # nodes initializations
        self.inpnodes=inputnodes
        self.hidnodes=hiddennodes
        self.outnodes=outputnodes

        # set mixing/learn rate
        self.mix=mixingfactor

        # set initial weights with a gaussian normal distribution centered at 0.0

        self.wih=numpy.random.normal(0.0, pow(self.hidnodes, -0.5),(self.hidnodes,self.inpnodes))
        self.who=numpy.random.normal(0.0, pow(self.outnodes, -0.5),(self.outnodes,self.hidnodes))


        pass        

    def train(self, input_list, target_list):

        input=numpy.array(input_list, ndmin=2).T
        target=numpy.array(target_list, ndmin=2).T

        # compute signals to hidden layer : Xhidden =Wih x Inp
        hidinp= numpy.dot(self.wih,input)
        # apply sigmoid function          : O_hidden=sigmoid(Xhidden)
        hidout= scipy.special.expit(hidinp)

        # compute signals to output layer : Xout =Who x Hout
        finalinp= numpy.dot(self.who, hidout)
        # apply sigmoid function          : O_final=sigmoid(Xout)
        finalout= scipy.special.expit(finalinp)

        # error : target - actual
        error_out = target - finalout

        # hiddenerrors : Whi^T x error_out
        error_hid = numpy.dot(self.who.T, error_out)

        # update the weights Who
        self.who += self.mix * numpy.dot((error_out * finalout * (1.0 -finalout)), hidout.T)

        # update the weights Wih

        self.wih += self.mix * numpy.dot((error_hid * hidout * (1.0 -hidout)), input.T)

        pass

    def inpoutput(self, input_list):

        input=numpy.array(input_list, ndmin=2).T

        # compute signals to hidden layer : Xhidden =Wih x Inp
        hidinp= numpy.dot(self.wih,input) 
        # apply sigmoid function          : O_hidden=sigmoid(Xhidden)
        hidout= scipy.special.expit(hidinp)

        # compute signals to output layer : Xout =Who x Hout
        finalinp= numpy.dot(self.who, hidout)
        # apply sigmoid function          : O_final=sigmoid(Xout)
        finalout= scipy.special.expit(finalinp)

        print (finalout)
        return finalout


# create instance of our neural network

n_input=784
n_hidden=250   # this ist just a random selection. Could be optimzed
n_output=10

mixfac=0.1     # this ist just a random selection. Could be optimzed

testnet=neuralnetwork(n_input,n_hidden,n_output,mixfac)

# train the neural network

# read training data set
traindata_file= open("./mnist_train_100.csv", 'r')
traindata_list= traindata_file.readlines()
traindata_file.close()

# set training generations : this ist just a random selection. Could be optimzed

generations= 5

for x in range(generations):

    # loop over all numbers in training data
    for record in traindata_list:
        # split by delimiter ','
        values = record.split(',')
        # scale and shift the inputs to ensure the data points are [0.01,0.99]
        input = (numpy.asfarray(values[1:])/255.0 * 0.99) + 0.01
        # create the target values : 0.01 for false and 0.99 for true
        target = numpy.zeros(n_output) +0.01
        target[int(values[0])] =0.99
        testnet.train(input,target)
        pass
    pass

# test the neural network

# read test data set

testdata_file= open("./mnist_test_10.csv", 'r')
testdata_list= testdata_file.readlines()
testdata_file.close()


n=0
score=0

# loop over all numbers in test data
for record in testdata_list:
    # split by delimiter ','
    values = record.split(',')
    # save the correct answer 
    reference = int(values[0])
    print(reference, "reference value")  
    # scale and shift the inputs to ensure the data points are [0.01,0.99]
    input = (numpy.asfarray(values[1:])/255.0 * 0.99) + 0.01
    # generate the outputs
    output= testnet.inpoutput(input)
    # grep the maximum value to determine the target the network has predicted
    target = numpy.argmax(output)
    print(target, "networks prediction")
    n +=1    
    if (target == reference):
        score+=1
    else: 
        pass    

    pass

print("Performance in percent", score/n*100)



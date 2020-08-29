import numpy as np
import random
import matplotlib.pyplot as plt
import trebuchet
import sys
from collections import deque

# Implementation of a fully connected network whose weights are updated through gradient descent
# Implementation was done from scratch to enable direct passing of gradient information from the differential equation solver to the neural network
# Gradient propagation was estimated using a finite difference method
# The dense network can be scaled arbitrarily with any number of inputs and hidden units
# Trebuchet parameters are normalized for increased stability during training
# Inputs are desired target, outputs are mass of the counterweight
# Angle can be changed, but by default, is preset


class NeuralNetwork:
    def __init__(self, layers):
        self.inputnum = layers[0]
        self.outputnum = layers[-1]
        self.layers = {}
        self.gradients = {}
        self.layeractivations = {}
        self.layerinputs = {}
        self.layerpreactiv = {} # Pre activation inputs from the hidden layers
        self.layerout = ''
        self.buildlayers(layers)
        self.outputscale = [30, 30]
        self.outputbase = [60, 30]
        self.targetbase = 240
        self.targetscale = 20 # target = targetbase + {-1,1}*targetscale
        self.lrn = 3e-3
        self.regterm = 3e-3
        self.Treb = trebuchet.Trebuchet()

        self.epochdeque = deque(maxlen=20)

    def epochdequeAvg(self):
        avg = 0
        for val in self.epochdeque:
            avg = avg + val
        avg = avg/len(self.epochdeque)
        return avg

    def buildlayers(self, layers):
        for i in range(len(layers)-1):
            name = 'h' + str(i)
            self.layers[name] = np.random.uniform(-1, 1, (layers[i]+1,layers[i+1]))
            self.gradients[name] = np.zeros((layers[i]+1,layers[i+1]))
            self.layeractivations[name] = np.zeros(layers[i])
        self.layerout = name

    def relu(self, x):
        return x * (x > 0)

    def drelu(self, x):
        return (x > 0) * 1

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def dsigmoid(self, x):
        return np.exp(-x)/(1+np.exp(-x))**2

    def _nameshift(self, key, inc):
        name, num = self._split(key)
        k = name + str(int(num)+inc)
        return k

    def _split(self, word):
        return [char for char in word]

    def _addbias(self, vector):
        vec_withbias = np.ones(vector.shape[0]+1)
        vec_withbias[1:] = vector
        return vec_withbias



    def feedforward(self, input):
        layerinput = np.array(input)
        self.layerinputs['h0'] = self._addbias(layerinput) # HOLDS A1, A2, A3
        # print("===============")
        for k,v in self.layers.items(): # keys are names of layers, values are weight matrices
            layerinput_withbias = self._addbias(layerinput)
            self.layerinputs[k] = layerinput_withbias
            if k != self.layerout:
                layeractiv = np.dot(layerinput_withbias, v)
                self.layerpreactiv[self._nameshift(k,1)] = layeractiv
                layeractiv = self.relu(layeractiv) # RELU OR SIGMOID
            else:
                layeractiv = self.sigmoid(np.dot(layerinput_withbias, v))
                layeractiv = self.outputscale[0]*layeractiv+self.outputbase[0]
            layerinput = layeractiv

        # print("BEGIN GRADIENT CALCULATION")
        loss, grad, range = self.getGradients_trebuchet(input, layeractiv)

        delta_curr = grad
        for keys in sorted(self.layers.keys(), reverse=True):
            activreshape = self.layerinputs[keys].reshape(-1,1)
            self.gradients[keys] = (delta_curr @ activreshape.transpose()).transpose()
            weights = self.layers[keys]
            self.gradients[keys][:, 1:] = self.gradients[keys][:, 1:] + self.regterm*weights[:, 1:]

            if keys == 'h0': break
            newdelta = np.dot(weights, delta_curr)
            deltapart = newdelta[1:]
            dactiv = self.drelu(self.layerpreactiv[keys]).reshape(-1,1) # D SIGMOID OR DRELU
            newdelta[1:] = deltapart*dactiv
            delta_curr = newdelta[1:]




        return loss, range, layeractiv

    def rangeloss(self, target, v, theta):
        range = v**2*np.sin(2*theta)/9.8 # g = 9.8 m/s, R = v^2*sin(2theta)/g
        # print("Range: {}".format(range))
        loss = (range - target)**2/2
        # print("Loss: {}".format(loss))
        dL_dv = (range - target)*2*v*np.sin(2*theta)/9.8*self.outputscale[0]
        dL_dtheta = (range - target)*v**2*np.cos(2*theta)/9.8*2*self.outputscale[1]

        grad = np.array([dL_dv, dL_dtheta])


        return loss, grad, range

    def rangeloss_est(self, target, v, theta, epsilon=1e-3):
        range = v**2*np.sin(2*theta)/9.8 # g = 9.8 m/s, R = v^2*sin(2theta)/g
        range_dv = (v+epsilon)**2*np.sin(2*theta)/9.8
        range_dtheta = v**2*np.sin(2*(theta+epsilon))/9.8

        loss = (range - target)**2/2
        dL_v = (range_dv - target)**2/2
        dL_theta = (range_dtheta - target)**2/2

        dL_dv = (dL_v-loss)/epsilon*self.outputscale[0]
        dL_dtheta = (dL_theta-loss)/epsilon*self.outputscale[1]



        grad = np.array([dL_dv, dL_dtheta])
        # print(grad)

        return loss, grad, range

    def getGradients_trebuchet(self, input, layeractiv, epsilon=1e-4):

        target = input # intended target
        target = self.scaleTarget(target, scaledown=False)
        # BASE
        self.Treb.resetStates()
        # self.Treb.setEnviroParam(input[0]) # Input 1 is the windspeed, input 2 is the target
        self.Treb.setControlParam(layeractiv, 45) # first output is for mW, second is for rAng
        range = self.Treb.runsim() # get the value of the estimated treb
        # First offset
        self.Treb.resetStates()
        self.Treb.setControlParam(layeractiv+epsilon, 45) # first output is for mW, second is for rAng
        range_first = self.Treb.runsim()
        self.Treb.resetStates()

        regloss = 0
        for key in self.layers.keys():
            regloss = regloss + self.sumWeights(key)
        regloss = self.regterm*regloss/2
        loss_base = (range-target)**2/2 + regloss
        loss_first = (range_first-target)**2/2 + regloss

        dL_dFirst = (loss_first-loss_base)*self.outputscale[0]
        grad = np.array([dL_dFirst])
        grad = grad.reshape(-1,1)
        return loss_base, grad, range


    def applyGradients(self):
        for k,v in self.layers.items():
            self.layers[k] = self.layers[k] - self.lrn*self.gradients[k]



    def printData(self, data):
        for k, v in data:
            print(k, v)
            print('*****')

    def scaleTarget(self, inputTarget, scaledown=True): # Scales target between 0 and 1
        inputTarget = np.array(inputTarget)
        if scaledown:
            target = (inputTarget - self.targetbase) / self.targetscale
            return target
        else:
            target = inputTarget*self.targetscale + self.targetbase
            return target

    def predict(self, input):
        layerinput = np.array(input)
        # layerinput_withbias = np.ones((layerinput.shape[0] + 1))
        # print(layerinput_withbias)
        # layerinput_withbias[1:] = layerinput
        self.layerinputs['h0'] = self._addbias(layerinput)  # HOLDS A1, A2, A3
        # print("===============")
        for k, v in self.layers.items():  # keys are names of layers, values are weight matrices
            layerinput_withbias = self._addbias(layerinput)
            self.layerinputs[k] = layerinput_withbias
            if k != self.layerout:
                layeractiv = np.dot(layerinput_withbias, v)
                self.layerpreactiv[self._nameshift(k, 1)] = layeractiv
                layeractiv = self.sigmoid(layeractiv)
            else:
                layeractiv = self.sigmoid(np.dot(layerinput_withbias, v))
                layeractiv = self.outputscale[0] * layeractiv + self.outputbase[0]
            layerinput = layeractiv

        _, grad, range = self.getGradients_trebuchet(input, layerinput)
        print(grad)
        return layerinput, range

    def sumWeights(self, key):
        return np.sum(np.square(self.layers[key][:, 1:]))

    def predictedUpdate(self, input):
        loss, pred, out = self.feedforward(input)
        self.applyGradients()
        return [pred, out]



layers = [1, 64, 256, 256, 64, 1]
NN = NeuralNetwork(layers)
epochs = []
losses = []

# NN.printData(NN.layers.items())

print("********")
out1, range1 = NN.predict([NN.scaleTarget(225)])
print("&"*20)
# print(out1)
out2, range2 = NN.predict([NN.scaleTarget(255)])
# print(out2)
print(range1, range2)
inputval = input("Press Enter to continue or q to quit: ")
if inputval == 'q':
    sys.exit()

for k,v in NN.layers.items():
    print(k, v.shape)

count = 0
totalepoch = 500
breakpointavg = 128
for e in range(totalepoch+1):
    # windspeed = random.randint(0, 5)
    windspeed = 0
    target = random.randint(NN.targetbase-NN.targetscale, NN.targetbase+NN.targetscale)
    target_scaled = NN.scaleTarget(target)
    # print(target_scaled)
    # target = 230
    # input_scaled = [windspeed, target_scaled] # 2 inputs to the system
    input_scaled = [target_scaled]
    loss, pred, out = NN.feedforward(input_scaled)
    i = 0
    # while (loss > 10) or (i < 20):
    # print(e, input)
    # for i in range(20):
    # loss, pred, out = NN.feedforward(input_scaled)
    NN.applyGradients()


    NN.epochdeque.append(loss)
    avg = NN.epochdequeAvg()
    if (avg < 40) and (NN.lrn <= 5e-5): break
    if (e)%5 == 0:
        print("EPOCH: {}, TARGET: {}, OUTPUT: {}, RANGE: {}, LOSS: {}".format(e, target, out, pred, avg))
        epochs.append(e)
        losses.append(NN.epochdequeAvg())
        #  count += 1
        # i += 1

    if avg < breakpointavg and NN.lrn > 5e-5:
        NN.lrn = NN.lrn/2
        if breakpointavg > 8:
            breakpointavg = breakpointavg/2
        else:
            breakpointavg = 8
        print("SETTING LEARNING RATE LOWER: {} {}".format(NN.lrn, breakpointavg))

print("********")
NNin = []
NNout = []
NNerr = []

soln1 = NN.predictedUpdate([NN.scaleTarget(228)])
soln2 = NN.predictedUpdate([NN.scaleTarget(237)])
soln3 = NN.predictedUpdate([NN.scaleTarget(243)])
soln4 = NN.predictedUpdate([NN.scaleTarget(254)])
print(soln1, soln2, soln3, soln4)

acc = 0
for _ in range(100):
    target = random.randint(NN.targetbase-NN.targetscale, NN.targetbase+NN.targetscale)
    target_scaled = NN.scaleTarget(target)
    input_scaled = [target_scaled]
    pred = NN.predictedUpdate([NN.scaleTarget(target)])
    err = (pred[0]-target)/pred[0]
    acc = acc + np.absolute(err)

print("Acc:{}".format(acc))


plt.figure(1)
plt.plot(epochs, losses)
plt.title('Training Loss v Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


print("DONE")

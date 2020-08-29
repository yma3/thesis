import numpy as np
import random
import matplotlib.pyplot as plt

# Implementation of a fully connected neural network on a Toy projectile motion example.
# Refer to Simplifiedtreb.py in the case of the trebuchet.

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
        self.outputscale = [20, np.pi/2]
        self.lrn = 0.0001

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
        for k,v in self.layers.items(): # keys are names of layers, values are weight matrices
            layerinput_withbias = self._addbias(layerinput)
            self.layerinputs[k] = layerinput_withbias
            if k != self.layerout:
                layeractiv = np.dot(layerinput_withbias, v)
                self.layerpreactiv[self._nameshift(k,1)] = layeractiv
                layeractiv = self.relu(layeractiv)
            else:
                layeractiv = self.sigmoid(np.dot(layerinput_withbias, v))
                layeractiv[0] = self.outputscale[0]*layeractiv[0]
                layeractiv[1] = self.outputscale[1]*layeractiv[1]
            layerinput = layeractiv

        # print("BEGIN GRADIENT CALCULATION")
        loss, grad, range = self.rangeloss(input, layeractiv[0], layeractiv[1])
        loss_est, grad_est, range_est = self.rangeloss_est(input, layeractiv[0], layeractiv[1])
        _, grad_est2, _ = self.rangeloss_est2(input, layeractiv[0], layeractiv[1])

        # print(loss, grad)
        delta_curr = grad_est
        for keys in sorted(self.layers.keys(), reverse=True):
            activreshape = self.layerinputs[keys].reshape(-1,1)
            self.gradients[keys] = (delta_curr @ activreshape.transpose()).transpose()
            if keys == 'h0': break
            weights = self.layers[keys]
            newdelta = np.dot(weights, delta_curr)
            deltapart = newdelta[1:]
            dactiv = self.drelu(self.layerpreactiv[keys]).reshape(-1,1)
            newdelta[1:] = deltapart*dactiv
            delta_curr = newdelta[1:]




        return loss, range, layeractiv

    def rangeloss(self, target, v, theta):
        range = v**2*np.sin(2*theta)/9.8 # g = 9.8 m/s, R = v^2*sin(2theta)/g
        loss = (range - target)**2/2
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

        return loss, grad, range

    def rangeloss_est2(self, target, v, theta, epsilon=1e-3):
        range = v**2*np.sin(2*theta)/9.8 # g = 9.8 m/s, R = v^2*sin(2theta)/g
        range_dv = (v+epsilon)**2*np.sin(2*theta)/9.8
        range_dtheta = v**2*np.sin(2*(theta+epsilon))/9.8

        loss = (range - target)**2/2
        dL_v = (range_dv - range)/epsilon
        dL_theta = (range_dtheta - range)/epsilon

        # print(range, dL_v, dL_theta)

        dL_dv = range*dL_v*self.outputscale[0]
        dL_dtheta = range*dL_theta*self.outputscale[1]



        grad = np.array([dL_dv, dL_dtheta])
        # print(grad)

        return loss, grad, range




    def applyGradients(self):
        for k,v in self.layers.items():
            # print("========")
            # print(v)
            self.layers[k] = self.layers[k] - self.lrn*self.gradients[k]
            # print(self.layers[k])
            # print("========")


    def printData(self, data):
        for k, v in data:
            print(k, v)
            print('*****')





layers = [1, 8, 8, 2]
NN = NeuralNetwork(layers)
epochs = []
losses = []

for k,v in NN.layers.items():
    print(k, v.shape)

for e in range(10000):
    target = random.randint(1, 10)
    target = [target]
    loss, pred, out = NN.feedforward(target)
    NN.applyGradients()
    if e%5 == 0:
        print("EPOCH: {}, TARGET: {}, OUTPUT: {}, RANGE: {}, LOSS: {}".format(e, target, out, pred, loss))
        epochs.append(e)
        losses.append(loss)


print(NN.feedforward([2]))
print(NN.feedforward([5]))
print(NN.feedforward([8]))

acc = 0
for _ in range(1000):
    target = random.randint(1, 10)
    target = [target]
    _, pred, _ = NN.feedforward(target)
    err = (target-pred)/target
    acc = acc + np.absolute(err)

acc = acc/1000*100
print(acc)

_, grad, _ = NN.rangeloss(1, 11, np.pi/4)
_, gradest, _ = NN.rangeloss_est(1, 11, np.pi/4)
_, gradest2, _ = NN.rangeloss_est2(1, 11, np.pi/4)

print(grad, gradest, gradest2)

_, grad, _ = NN.rangeloss(2, 4, np.pi/3)
_, gradest, _ = NN.rangeloss_est(2, 4, np.pi/3)
_, gradest2, _ = NN.rangeloss_est2(2, 4, np.pi/3)

print(grad, gradest, gradest2)

_, grad, _ = NN.rangeloss(2, 8, np.pi/6)
_, gradest, _ = NN.rangeloss_est(2, 8, np.pi/6)
_, gradest2, _ = NN.rangeloss_est2(2, 8, np.pi/6)

print(grad, gradest, gradest2)

_, grad, _ = NN.rangeloss(7, 7, 2*np.pi/7)
_, gradest, _ = NN.rangeloss_est(7, 7, 2*np.pi/7)
_, gradest2, _ = NN.rangeloss_est2(7, 7, 2*np.pi/7)

print(grad, gradest, gradest2)

fig = plt.figure(1)
plt.plot(epochs,losses)
plt.title('Loss vs Epochs')
plt.xlabel('Epoch No.')
plt.ylabel('Loss')
plt.show()


print("DONE")

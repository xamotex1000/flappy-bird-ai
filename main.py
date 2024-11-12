import math
import random
from test import *
import time
import os
from tkinter import *
import threading
bird = FlappyBird()
class Neuron:
    def __init__(self, weights):
        self.weights = weights
    def sigmoid(self, x):
        return 1/(1+math.exp(-x))
    def feedforward(self, inputs):
        weighted_sum = sum([inputs[i]*self.weights[i] for i in range(len(inputs))])
        output = self.sigmoid(weighted_sum)
        return(output)
class NeuralNetwork:
    def __init__(self, inputs, x, y):
        self.Q = []
        self.epsilon = 1.5
        self.layer = []
        for i in range(x):
            self.layer.append([])
            for j in range(y):
                weights = []
                if i == 0:
                    for k in range(inputs):
                        weights.append(random.uniform(-1,1))
                else:
                    for k in range(y):
                        weights.append(random.uniform(-1,1))
                self.layer[i].append(Neuron(weights))
        self.learning_rate = 0.01
    def update(self, state, action):
        if (state, action) not in self.Q:
            self.Q.append((state, action))
    def feedforward(self, layer, inputs):
        if layer == 0:
            if random.random() < self.epsilon:
                result = round(random.random())
                self.epsilon = max(0.1, self.epsilon*0.995)
            else:
                result = round(self.layer[layer][0].feedforward(inputs) + random.uniform(-0.1, 0.1))
            return [result]
        if layer < len(self.layer):
            result = self.feedforward(layer+1, [neuron.feedforward(inputs) for neuron in self.layer[layer]])
            return result
        else:
            return inputs
    def calculate_mse(self, predictions, targets, r=0, p=0):
        n = len(predictions)
        mse = 1/n*sum([(predictions[i]-targets[i])**2-r+p for i in range(n)])
        return mse
    def punish(self, inputs, reward):
        os.system("clear")
        predictions = self.feedforward(0, inputs)
        if round(predictions[0]) == 1:
            targets = [0]
        else:
            targets = [1]
        self.update(inputs, targets)
        for layer in range(len(self.layer)):
            for neuron in range(len(self.layer[layer])):
                for weight in range(len(self.layer[layer][neuron].weights)):
                    self.layer[layer][neuron].weights[weight]-=self.learning_rate*(self.calculate_mse(predictions, targets, 0, -reward)/self.layer[layer][neuron].weights[weight])
    def reward(self, inputs, reward):
        if(reward > 5):
            print("B")
        predictions = self.feedforward(0, inputs)
        targets = [round(predictions[0])]
        self.update(inputs, targets)
        for layer in range(len(self.layer)):
            for neuron in range(len(self.layer[layer])):
                for weight in range(len(self.layer[layer][neuron].weights)):
                    self.layer[layer][neuron].weights[weight]-=self.learning_rate*(self.calculate_mse(predictions, targets, reward, 0)/self.layer[layer][neuron].weights[weight])
    def calculate_gradient(self, x, y, z, inputs, targets):
        predictions = self.feedforward(0,inputs)
        old = self.calculate_mse(predictions,targets)
        weight = self.layer[x][y].weights[z]
        self.layer[x][y].weights[z] = weight+0.1
        predictions = self.feedforward(0,inputs)
        new = self.calculate_mse(predictions,targets)
        gradient = (new-old)/0.001
        weight -= self.learning_rate*gradient
        self.layer[x][y].weights[z] = weight
    def train(self, targets, inputs):
        predictions = self.feedforward(0,inputs)
        mse = self.calculate_mse(predictions, targets)
        for i in range(len(self.layer)):
            for j in range(len(self.layer[i])):
                for k in range(len(self.layer[i][j].weights)):
                    self.calculate_gradient(i,j,k,inputs,targets)
        return(self.feedforward(0,inputs))
"""
run_network = NeuralNetwork(5, 1, 1)


def run():
    training = [[[1,0,0,0,0],[0]],[[0,1,0,0,0],[0]],[[0,0,1,0,0],[1]],[[0,0,0,1,0],[0]],[[0,0,0,0,1],[0]]]
    for i in range(100000):
        rand = random.randint(0,4)
        run_network.train(training[rand][1],training[rand][0],0.01)

    accuracy = 0
    amt = 0
    for i in range(20):
        grid = [0,0,0,0,0]
        obstacle = random.randint(0,4)
        grid[obstacle] = 1
        output = run_network.feedforward(0,grid)
        if round(output[0]) == 1:
            amt+=1
            if obstacle == 2:
                accuracy+=1
    print(f"Accuracy: {accuracy/amt*100}%")
run()
"""

flap_network = NeuralNetwork(3,6,1)
thread = threading.Thread(target=keyloop, args=[bird,flap_network])
thread.start()

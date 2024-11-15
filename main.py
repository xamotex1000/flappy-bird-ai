import math
import random
from test import *
import time
import os
from tkinter import *
import threading
import pickle
class Neuron:
    def __init__(self, weights):
        self.weights = weights
    def sigmoid(self, x):
        if x > 50:
            return 1.0
        if x < -50:
            return 0.0
        return 1/(1+math.exp(-x))
    def feedforward(self, inputs):
        weighted_sum = sum([inputs[i]*self.weights[i] for i in range(len(inputs))])
        output = self.sigmoid(weighted_sum)
        return(output)
class NeuralNetwork:
    def __init__(self, inputs, x, y, outputs):
        self.Q = []
        self.epsilon = 1.0
        self.layer = []
        self.layer.append([])
        for i in range(inputs):
            weights = []
            for i in range(inputs):
                weights.append(random.uniform(-1,1))
            self.layer[0].append(Neuron(weights))
        for i,amt in enumerate(y):
            self.layer.append([])
            for j in range(amt):
                weights = []
                if i == 0:
                    for k in range(inputs):
                        weights.append(random.uniform(-1,1))
                else:
                    for k in range(y[i-1]):
                        weights.append(random.uniform(-1,1))
                self.layer[i+1].append(Neuron(weights))
        self.layer.append([])
        for i in range(outputs):
            weights = []
            for j in range(y[-1]):
                weights.append(random.uniform(-1,1))
            self.layer[-1].append(Neuron(weights))
        self.learning_rate = 0.05
    def save_weights(self, file_path="weights"):
        weights = [[neuron.weights for neuron in layer] for layer in self.layer]
        with open(file_path, 'wb') as file:
            pickle.dump(weights, file)
        print(f"Weights saved to {file_path}")
    def load_weights(self, file_path="weights"):
        with open(file_path, 'rb') as file:
            weights = pickle.load(file)
        for i, layer in enumerate(self.layer):
            for j, neuron in enumerate(layer):
                neuron.weights = weights[i][j]
        print(f"Weights loaded from {file_path}")
    def q_learning_update(self, state, action, reward, next_state, gamma=0.95):
        q_values = self.feedforward(state)
        next_q_values = self.feedforward(next_state)
        target = reward+gamma*max(next_q_values)
        error = target-q_values[0]
        self.train([target if i == action else q_values[i] for i in range(len(q_values))], state)
    def feedforward(self, inputs):
        current_inputs = inputs
        for layer in self.layer:
            next_inputs = []
            for neuron in layer:
                next_inputs.append(neuron.feedforward(current_inputs))
            current_inputs = next_inputs
        return current_inputs
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0,1)
        else:
            q_values = self.feedforward(state)
            return q_values.index(max(q_values))
    def calculate_mse(self, predictions, targets, r=0, p=0):
        n = len(predictions)
        mse = 1/n*sum([(predictions[i]-targets[i])**2-r+p for i in range(n)])
        return mse
    def punish(self, state, action, penalty, next_state):
        self.q_learning_update(state, action, penalty, next_state)
    def reward(self, state, action, reward, next_state):
        self.q_learning_update(state, action, reward, next_state)
    def calculate_gradient(self, layer_index, neuron_index, weight_index, inputs, targets):
        activations = [inputs]
        current_inputs = inputs
        for layer in self.layer:
            layer_outputs = []
            for neuron in layer:
                layer_outputs.append(neuron.feedforward(current_inputs))
            activations.append(layer_outputs)
            current_inputs = layer_outputs
        layer_deltas = [0] * len(self.layer)
        for i in reversed(range(len(self.layer))):
            layer_deltas[i] = []
            for j, neuron in enumerate(self.layer[i]):
                if i == len(self.layer)-1:
                    error = activations[i+1][j]-targets[j]
                else:
                    error = sum(layer_deltas[i+1][k]*self.layer[i+1][k].weights[j] for k in range(len(self.layer[i+1])))
                delta = error*neuron.sigmoid(activations[i+1][j])*(1-neuron.sigmoid(activations[i+1][j]))
                layer_deltas[i].append(delta)
        neuron = self.layer[layer_index][neuron_index]
        gradient = layer_deltas[layer_index][neuron_index]*activations[layer_index][weight_index]
        neuron.weights[weight_index]-=self.learning_rate*gradient
    def train(self, targets, inputs):
        predictions = self.feedforward(inputs)
        mse = self.calculate_mse(predictions, targets)
        for i in range(len(self.layer)):
            for j in range(len(self.layer[i])):
                for k in range(len(self.layer[i][j].weights)):
                    self.calculate_gradient(i,j,k,inputs,targets)
        return(mse)
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

flap_network = NeuralNetwork(3,3,[3,8,16,16],1)
keyloop(flap_network)

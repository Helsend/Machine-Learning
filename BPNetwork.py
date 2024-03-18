"""
name:BPNetwork
author:
"""
import random
import numpy as np


def sigmoid(z):
    """sigmoid函数"""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """sigmoid函数的导函数"""
    return sigmoid(z)*(1-sigmoid(z))


class BPNetwork(object):
    def __init__(self, sizes):
        """
        function:定义网络结构
        num_layers:网络层数
        sizes:网络结构
        biases:网络偏置
        weights:网络权重
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """返回网络输出"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def accuracy(self, test_data):
        """返回神经网络在test_data上预测的accuracy"""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)/len(test_data)

    def train_accuracy(self, training_data):
        """返回神经网络在training_data上预测的accuracy"""
        train_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in training_data]
        return sum(int(x == y) for (x, y) in train_results) / len(training_data)

    def backprop(self, x, y):
        """根据一个样本计算并返回w和b的梯度"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def update_mini_batch(self, mini_batch, eta):
        """根据mini_batch个样本的self.backprop(x, y)结果计算w和b的梯度，并更新w和b"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data):
        """使用小批量随机梯度下降训练神经网络"""
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            print("Epoch {0}:training set accuracy:{1} ; test set accuracy:{2};".format(j, self.train_accuracy(training_data), self.accuracy(test_data)))



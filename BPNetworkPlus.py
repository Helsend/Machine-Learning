"""
name:BPNetworkPlus
author:
对BPNetwork进行了如下改进：
1、可以从交叉熵损失函数和二次代价函数中选一个作为损失函数
2、使用均值为0，方差为1/n的高斯分布初始化权重（而不是均值为0，方差为1的高斯分布）
3、采用L2规范化
"""

import random
import numpy as np


def sigmoid(z):
    """sigmoid函数"""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """sigmoid函数的导函数"""
    return sigmoid(z)*(1-sigmoid(z))


class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        """返回损失"""
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """从输出层返回误差增量"""
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        """返回损失"""
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """从输出层返回误差增量"""
        return a-y


class BPNetworkPlus(object):
    def __init__(self, sizes, cost=CrossEntropyCost):
        """
        :param sizes: 网络结构
        :param cost: 损失函数类
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """返回网络输出"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

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
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_b, nabla_w

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """
        function:根据mini_batch个样本的self.backprop(x, y)结果计算w和b的梯度，并更新w和b
        :param mini_batch: 小批量训练集，元组列表(x,y)
        :param eta:学习率
        :param lmbda:正则化参数
        :param n:训练集大小
        :return:
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def sgd(self, training_data, validation_data, epochs, mini_batch_size, eta, lmbda=0.0):
        """
        function:使用小批量随机梯度下降训练神经网络
        :param training_data: 训练集
        :param validation_data: 验证集
        :param epochs: 迭代次数
        :param mini_batch_size: 小批量大小
        :param eta: 学习率
        :param lmbda: 正则化参数
        :return:
        """
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)
            print("Epoch {0}:training set accuracy:{1} ; test set accuracy:{2};".format(j, self.train_accuracy(training_data), self.accuracy(validation_data)))

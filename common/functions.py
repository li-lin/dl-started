import numpy as np


def testting(a,b):
    return a+b

# 恒等函数
def identify_function(x):
    return x

# 阶跃函数
def step_function(x):
    return np.array(x > 0, dtype=np.int)

# sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# relu函数
def relu(x):
    return np.maximum(0, x)

# softmax函数
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x -= np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x -= np.max(x) # overflow guard
    return np.exp(x) / np.sum(np.exp(x))

# 均方误差函数
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

# 交叉熵误差
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

# sigmoid函数的梯度
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
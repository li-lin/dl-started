class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        # 会将out中对应于self.mask中True的位置的所有元素都置为0。
        out[self.mask] = 0

        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
    
# Sigmoid层的实现
import numpy as np

class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

# Affine层的实现
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T) # T表示W的转置
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
    
# Softmax-with-loss层的实现
from common.functions import cross_entropy_error, softmax

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 损失
        self.y = None # softmax的输出
        self.t = None # 目标值

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

# test
# import numpy as np

# x = np.array([[1.0, -0.5], [-2.0, 3.0]])
# print(x)

# mask = (x <= 0)
# print(mask)

# relu = Relu()
# out = relu.forward(x)
# print(out)

# x = np.array([[-1.0, 2.0, 3.0, -4.0], [4.0, 5.0, -6.0, -7.0]])
# mask = (x <= 0)
# print(mask)
# x[mask]=0   
# print(x)
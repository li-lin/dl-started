import numpy as np

# 假设我们有一些简单的训练数据（面积和房价）
X = np.array([1000, 1500, 2000])  # 输入特征（面积）
y = np.array([300000, 450000, 600000])  # 真实值（房价）

# 模型的初始参数（权重和偏置）
w = 0.1
b = 0.0

# 学习率
learning_rate = 0.0000001

# 训练过程
for epoch in range(100):
    # 预测房价
    y_pred = w * X + b  # 线性模型的预测值
    
    # 计算损失（均方误差）
    loss = np.mean((y - y_pred) ** 2)
    
    # 打印当前轮次和损失值
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')
    
    # 计算梯度（损失函数对w和b的偏导数）
    dw = -2 * np.mean(X * (y - y_pred))
    db = -2 * np.mean(y - y_pred)
    
    # 使用梯度更新参数
    w -= learning_rate * dw
    b -= learning_rate * db

print(f'Trained parameters: w = {w}, b = {b}')

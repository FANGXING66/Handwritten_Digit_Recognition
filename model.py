import torch  # Import PyTorch
import torch.nn as nn  # Import the neural network module from PyTorch

# Define the neural network class, inheriting from nn.Module
class Network(nn.Module):
    """
    神经网络模型类
    使用简单的全连接层结构
    输入层784节点（28x28像素）
    隐藏层256节点
    输出层10节点（0-9数字）
    """
    def __init__(self):
        super().__init__()  # Call the initializer of the parent class nn.Module
        # 定义两个全连接层
        self.layer1 = nn.Linear(784, 256)  # 输入层到隐藏层
        self.layer2 = nn.Linear(256, 10)   # 隐藏层到输出层

    def forward(self, x):
        """
        前向传播函数
        参数:
            x: 输入张量，形状为[batch_size, 1, 28, 28]
        返回:
            输出张量，形状为[batch_size, 10]
        """
        x = x.view(-1, 28*28)  # 将图像展平为一维向量
        x = self.layer1(x)     # 第一层线性变换
        x = torch.relu(x)      # ReLU激活函数
        return self.layer2(x)  # 第二层线性变换

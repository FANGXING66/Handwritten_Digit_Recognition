# 手写数字识别项目

## 快速开始
```bash
# 安装依赖
pip install -r requirements.txt

# 准备数据集
python parse_train_images_labels.py
python parse_t10k_images_labels.py

# 训练模型
python model_train.py

# 运行GUI程序
python gui_recognition.py
```

## 项目概述
这是一个基于PyTorch的手写数字识别项目，使用MNIST数据集进行训练。项目包含完整的训练和推理流程，以及一个简单的GUI界面用于实时识别手写数字。

### 项目结构
```
├── MNIST_data/          # 原始MNIST数据集
├── mnist_train/         # 处理后的训练集
├── mnist_test/          # 处理后的测试集
├── model.py            # 模型定义
├── model_train.py      # 模型训练
├── model_inference.py  # 模型测试
├── gui_recognition.py  # GUI界面程序
├── parse_train_images_labels.py  # 训练集处理
└── parse_t10k_images_labels.py   # 测试集处理
```

### 文件说明

#### model.py
- 定义神经网络模型结构
- 包含一个简单的全连接神经网络
- 输入层784节点，隐藏层256节点，输出层10节点

#### model_train.py
- 负责模型的训练过程
- 使用Adam优化器和交叉熵损失函数
- 训练10个epoch，批量大小为64
- 训练完成后保存模型到mnist.pth

#### model_inference.py
- 用于评估模型性能
- 在测试集上进行预测
- 输出模型准确率和错误案例

#### gui_recognition.py
- 提供图形用户界面
- 支持手写数字输入
- 实时显示识别结果

#### parse_train_images_labels.py
- 处理MNIST训练集数据
- 将原始数据转换为图像文件
- 按数字类别组织存储

#### parse_t10k_images_labels.py
- 处理MNIST测试集数据
- 将原始数据转换为图像文件
- 按数字类别组织存储

### 模型架构
```python
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 256)  # 输入层到隐藏层
        self.layer2 = nn.Linear(256, 10)   # 隐藏层到输出层

    def forward(self, x):
        x = x.view(-1, 28*28)  # 将图像展平
        x = self.layer1(x)     # 第一层线性变换
        x = torch.relu(x)      # ReLU激活函数
        return self.layer2(x)  # 第二层线性变换
```

### 训练过程
1. 数据预处理：
   - 转换为灰度图
   - 转换为张量格式

2. 训练参数：
   - 优化器：Adam
   - 损失函数：交叉熵损失
   - 训练轮数：10
   - 批量大小：64

3. 训练效果：
   - 测试集准确率：97.77%

### 使用方法
1. 准备环境：
   ```bash
   pip install -r requirements.txt
   ```

2. 准备数据：
   ```bash
   python parse_train_images_labels.py
   python parse_t10k_images_labels.py
   ```

3. 训练模型：
   ```bash
   python model_train.py
   ```

4. 运行GUI程序：
   ```bash
   python gui_recognition.py
   ```

### 依赖项
- PyTorch
- torchvision
- PIL (Python Imaging Library)
- numpy
- tkinter (GUI界面)

### 注意事项
1. 确保MNIST数据集已正确放置在MNIST_data目录下
2. 训练过程需要一定时间，请耐心等待
3. 使用GUI程序时，建议：
   - 写数字时尽量写大一些
   - 保持笔画清晰完整
   - 将数字写在画布中央

### 许可证
本项目采用MIT许可证 - 详见LICENSE文件

### 致谢
- MNIST数据集由Yann LeCun提供
- 项目使用PyTorch深度学习框架
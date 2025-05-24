import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import Network

# 设置随机种子，确保结果可复现
torch.manual_seed(42)

# 定义数据转换
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
    transforms.ToTensor()  # 转换为张量
])

# 加载训练集和测试集
train_dataset = datasets.ImageFolder(root='./mnist_train', transform=transform)
test_dataset = datasets.ImageFolder(root='./mnist_test', transform=transform)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 创建模型实例
model = Network()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters())  # Adam优化器

# 训练模型
def train(epochs):
    """
    训练模型的主函数
    参数:
        epochs: 训练轮数
    """
    for epoch in range(epochs):
        model.train()  # 设置为训练模式
        running_loss = 0.0
        
        # 遍历训练数据
        for i, (images, labels) in enumerate(train_loader):
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 每100个批次打印一次损失
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # 每个epoch结束后打印平均损失
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {running_loss/len(train_loader):.4f}')

# 测试模型
def test():
    """
    在测试集上评估模型性能
    """
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'测试集准确率: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    # 打印数据集信息
    print(f'训练集大小: {len(train_dataset)}')
    print(f'测试集大小: {len(test_dataset)}')
    
    # 训练模型
    print('开始训练...')
    train(epochs=10)
    
    # 测试模型
    print('开始测试...')
    test()
    
    # 保存模型
    torch.save(model.state_dict(), 'mnist.pth')
    print('模型已保存到 mnist.pth')

import torch
from torchvision import datasets, transforms
from model import Network

# 定义数据转换
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
    transforms.ToTensor()  # 转换为张量
])

# 加载测试数据集
test_dataset = datasets.ImageFolder(root='./mnist_test', transform=transform)
print(f'测试集大小: {len(test_dataset)}')

# 创建模型实例并加载训练好的权重
model = Network()
model.load_state_dict(torch.load('mnist.pth'))
model.eval()  # 设置为评估模式

# 统计变量
correct = 0
total = 0

# 遍历测试集进行预测
with torch.no_grad():
    for images, labels in test_dataset:
        # 添加批次维度
        images = images.unsqueeze(0)
        
        # 进行预测
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        # 更新统计信息
        total += 1
        if predicted.item() == labels:
            correct += 1
        else:
            # 打印错误预测的案例
            print(f'预测错误: 预测值 = {predicted.item()} 实际值 = {labels} 图片路径 = {test_dataset.imgs[total-1][0]}')

# 打印最终准确率
print(f'测试准确率 = {correct} / {total} = {correct/total}')

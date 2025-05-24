import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageTk, ImageDraw, ImageOps
import torch.nn as nn
import torch.nn.functional as F
from model import Network

class DrawingApp:
    """
    手写数字识别应用程序
    提供画布供用户绘制数字，并使用训练好的模型进行识别
    """
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别")
        
        # 创建画布
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack(pady=20)
        
        # 创建按钮
        self.recognize_button = tk.Button(root, text="识别", command=self.recognize_digit)
        self.recognize_button.pack(side=tk.LEFT, padx=20)
        
        self.clear_button = tk.Button(root, text="清除", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=20)
        
        # 创建结果显示标签
        self.result_label = tk.Label(root, text="", font=('Arial', 24))
        self.result_label.pack(pady=20)
        
        # 初始化绘图变量
        self.last_x = None
        self.last_y = None
        self.line_width = 15  # 线条宽度
        
        # 绑定鼠标事件
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)
        
        # 加载模型
        self.model = Network()
        self.model.load_state_dict(torch.load('mnist.pth'))
        self.model.eval()
        
        # 定义图像转换
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

    def start_draw(self, event):
        """开始绘制"""
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        """绘制线条"""
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                  width=self.line_width, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)
        self.last_x = event.x
        self.last_y = event.y

    def stop_draw(self, event):
        """停止绘制"""
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        """清除画布"""
        self.canvas.delete("all")
        self.result_label.config(text="")

    def recognize_digit(self):
        """识别手写数字"""
        # 创建新图像
        image = Image.new('RGB', (280, 280), 'white')
        draw = ImageDraw.Draw(image)
        
        # 获取画布上的所有线条并绘制到新图像上
        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            draw.line(coords, fill='black', width=self.line_width)
        
        # 调整图像大小并转换为灰度图
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        image = ImageOps.invert(image)  # 反转颜色，使数字为白色，背景为黑色
        
        # 转换为张量并进行预测
        image_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted].item()
        
        # 显示结果
        self.result_label.config(text=f"预测结果: {predicted}\n置信度: {confidence:.2%}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop() 
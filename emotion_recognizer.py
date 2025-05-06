import torch
import torch.nn as nn
import cv2
import numpy as np
from models.vgg import VGG
import torchvision.transforms as transforms

class EmotionRecognizer:
    def __init__(self, model_path='FER2013_VGG19/PrivateTest_model.t7'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载模型
        self.model = VGG('VGG19')
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['net'])
        self.model.to(self.device)
        self.model.eval()
        
        # 定义表情标签
        self.emotions = ['生气', '厌恶', '恐惧', '开心', '伤心', '惊讶', '平静']
        
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def preprocess_face(self, face_img):
        """预处理人脸图像"""
        # 转换为灰度图
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
            
        # 调整大小为48x48
        resized = cv2.resize(gray, (48, 48))
        
        # 转换为PyTorch张量
        tensor = torch.from_numpy(resized).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
        tensor = tensor.repeat(1, 3, 1, 1)  # 复制到3个通道
        tensor = tensor / 255.0  # 归一化
        tensor = (tensor - 0.5) / 0.5  # 标准化
        
        return tensor
    
    def predict_emotion(self, face_img):
        """预测人脸表情"""
        try:
            # 预处理图像
            tensor = self.preprocess_face(face_img)
            tensor = tensor.to(self.device)
            
            # 进行预测
            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(probs, dim=1)
                confidence = float(probs[0][predicted])
                
            emotion = self.emotions[predicted]
            return {
                'emotion': emotion,
                'confidence': confidence,
                'success': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            } 
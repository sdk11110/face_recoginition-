import cv2
import numpy as np
import logging
from models.face_recognition import FaceRecognizer
import os

def process_face_image(image_data):
    """处理人脸图像，返回编码和裁剪后的人脸图像"""
    try:
        # 将字节数据转换为numpy数组
        nparr = np.frombuffer(image_data, np.uint8)
        # 解码图像
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logging.error("无法解码图像")
            return None, None
            
        # 获取人脸识别器实例
        face_recognizer = FaceRecognizer()
        
        # 检测人脸
        faces = face_recognizer.detect_faces(image)
        if not faces:
            logging.error("未检测到人脸")
            return None, None
            
        # 获取第一个人脸
        face = faces[0]
        
        # 裁剪人脸区域
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        face_image = image[top:bottom, left:right]
        
        # 获取人脸编码
        face_encoding = face_recognizer.get_face_encoding(image, face)
        
        return face_encoding, face_image
        
    except Exception as e:
        logging.error(f"处理人脸图像时出错: {str(e)}")
        return None, None

def save_face_image(face_image, user_id):
    """保存人脸图像"""
    try:
        # 确保目录存在
        os.makedirs('static/faces', exist_ok=True)
        
        # 保存图像
        file_path = f'static/faces/{user_id}.jpg'
        cv2.imwrite(file_path, face_image)
        
        return file_path
    except Exception as e:
        logging.error(f"保存人脸图像时出错: {str(e)}")
        return None 
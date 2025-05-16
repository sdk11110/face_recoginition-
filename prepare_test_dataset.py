import os
import sys
import shutil
import cv2
import dlib
import numpy as np
import argparse
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dlib 人脸检测器
detector = dlib.get_frontal_face_detector()

def extract_face(image_path, target_size=(150, 150)):
    """
    从图像中提取人脸
    参数:
        image_path: 图像路径
        target_size: 目标大小(宽, 高)
    返回:
        成功返回处理后的人脸图像，失败返回None
    """
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"无法读取图像: {image_path}")
            return None
            
        # 转换为RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 检测人脸
        faces = detector(rgb_img, 1)
        if not faces:
            logger.warning(f"未检测到人脸: {image_path}")
            return None
            
        # 处理第一个人脸
        face = faces[0]
        
        # 裁剪人脸区域
        face_img = img[face.top():face.bottom(), face.left():face.right()]
        
        # 调整大小
        face_img_resized = cv2.resize(face_img, target_size)
        
        return face_img_resized
        
    except Exception as e:
        logger.error(f"处理图像 {image_path} 时出错: {e}")
        return None

def prepare_test_dataset(source_dir, target_dir, n_samples=10):
    """
    准备测试数据集
    参数:
        source_dir: 源目录，包含多个人的文件夹
        target_dir: 目标目录
        n_samples: 每个人要提取的样本数量
    """
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 处理源目录中的每个人
    for person_name in os.listdir(source_dir):
        person_dir = os.path.join(source_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        logger.info(f"处理 {person_name} 的图像...")
        
        # 创建目标人员目录
        target_person_dir = os.path.join(target_dir, person_name)
        os.makedirs(target_person_dir, exist_ok=True)
        
        # 获取所有图像文件
        image_files = [f for f in os.listdir(person_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                      
        if not image_files:
            logger.warning(f"{person_name} 目录中没有图像文件")
            continue
            
        # 限制样本数量
        sample_count = min(n_samples, len(image_files))
        selected_images = image_files[:sample_count]
        
        # 处理每个选中的图像
        successful_count = 0
        for i, img_file in enumerate(selected_images):
            img_path = os.path.join(person_dir, img_file)
            
            # 提取人脸
            face_img = extract_face(img_path)
            if face_img is not None:
                # 保存处理后的人脸
                target_path = os.path.join(target_person_dir, f"{i+1}.jpg")
                cv2.imwrite(target_path, face_img)
                successful_count += 1
                
        logger.info(f"已为 {person_name} 准备 {successful_count} 张测试图像")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="准备人脸识别测试数据集")
    parser.add_argument("--source", "-s", required=True, help="源图像目录")
    parser.add_argument("--target", "-t", default="data/test_faces", help="目标测试数据集目录")
    parser.add_argument("--samples", "-n", type=int, default=10, help="每个人的样本数量")
    
    args = parser.parse_args()
    
    # 准备测试数据集
    prepare_test_dataset(args.source, args.target, args.samples)
    
    logger.info(f"测试数据集准备完成，保存在 {args.target}")
    logger.info("您可以使用系统中的评估功能来测试人脸识别性能")

if __name__ == "__main__":
    main() 
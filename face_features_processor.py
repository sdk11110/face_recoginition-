# Extract features from images and save into "features_all.csv"

import os
import dlib
import csv
import numpy as np
import logging
import cv2
import mysql.connector
import face_recognition

# MySQL配置
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',  # 确保这是你的MySQL密码
    'database': 'attendance_db'
}

def get_db_connection():
    return mysql.connector.connect(**db_config)

#  Path of cropped faces
path_images_from_camera = "data/data_faces_from_camera/"

#  Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

#  Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/data_dlib/shape_predictor_68_face_landmarks.dat')

#  Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

#  Return 128D features for single image

def return_128d_features(path_img):
    img_rd = cv2.imread(path_img)
    faces = detector(img_rd, 1)

    logging.info("%-40s %-20s", " Image with faces detected:", path_img)

    # For photos of faces saved, we need to make sure that we can detect faces from the cropped images
    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    else:
        face_descriptor = 0
        logging.warning("no face")
    return face_descriptor


#   Return the mean value of 128D face descriptor for person X

def return_features_mean_personX(path_face_personX):
    features_list_personX = []
    photos_list = os.listdir(path_face_personX)
    
    if not photos_list:
        logging.error(f"错误：文件夹 {path_face_personX} 中没有照片！")
        return None
        
    valid_photos = 0
    for i in range(len(photos_list)):
        #  return_128d_features()  128D  / Get 128D features for single image of personX
        logging.info("%-40s %-20s", " / Reading image:", path_face_personX + "/" + photos_list[i])
        features_128d = return_128d_features(path_face_personX + "/" + photos_list[i])
        #  Jump if no face detected from image
        if features_128d == 0:
            logging.warning(f"警告：在照片 {photos_list[i]} 中未检测到人脸")
            continue
        else:
            features_list_personX.append(features_128d)
            valid_photos += 1
            
    if valid_photos == 0:
        logging.error(f"错误：在文件夹 {path_face_personX} 中没有找到有效的人脸照片！")
        return None
        
    if features_list_personX:
        features_mean_personX = np.array(features_list_personX, dtype=object).mean(axis=0)
        logging.info(f"成功处理 {valid_photos} 张有效照片")
    else:
        features_mean_personX = np.zeros(128, dtype=object, order='C')
    return features_mean_personX

def save_to_database(username, features_mean, user_id):
    """
    保存人脸特征到数据库
    :param username: 用户名
    :param features_mean: 人脸特征向量
    :param user_id: 用户ID
    """
    conn = None
    cursor = None
    try:
        # 确保user_id是整数
        user_id = int(user_id)
        
        # 检查参数
        if not username or not isinstance(features_mean, np.ndarray):
            raise ValueError("无效的参数：用户名或特征向量为空")
            
        # 检查特征向量的维度
        if features_mean.shape[0] != 128:
            raise ValueError(f"特征向量维度错误：期望128，实际{features_mean.shape[0]}")
            
        # 获取数据库连接
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 将特征向量转换为二进制数据
        face_encoding_binary = features_mean.tobytes()
        
        # 首先检查是否已存在该用户的人脸数据
        cursor.execute("SELECT id FROM known_faces WHERE user_id = %s", (user_id,))
        existing_record = cursor.fetchone()
        
        if existing_record:
            # 更新现有记录
            cursor.execute("""
                UPDATE known_faces 
                SET face_encoding = %s, 
                    name = %s,
                    is_active = TRUE,
                    created_at = CURRENT_TIMESTAMP
                WHERE user_id = %s
            """, (face_encoding_binary, username, user_id))
            logging.info(f"更新用户 {username} (ID: {user_id}) 的人脸特征")
        else:
            # 获取下一个可用的最小ID
            cursor.execute("SELECT id FROM known_faces ORDER BY id")
            used_ids = [row[0] for row in cursor.fetchall()]
            
            next_id = 1
            for current_id in sorted(used_ids):
                if current_id != next_id:
                    break
                next_id += 1
            
            # 插入新记录
            cursor.execute("""
                INSERT INTO known_faces (id, user_id, name, face_encoding, is_active)
                VALUES (%s, %s, %s, %s, TRUE)
            """, (next_id, user_id, username, face_encoding_binary))
            logging.info(f"插入用户 {username} (ID: {user_id}) 的人脸特征")
        
        # 提交事务
        conn.commit()
        logging.info(f"成功保存用户 {username} (ID: {user_id}) 的人脸特征到数据库")
        
    except ValueError as ve:
        logging.error(f"参数错误: {str(ve)}")
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        error_msg = f"保存人脸特征到数据库时出错: {str(e)}"
        logging.error(error_msg)
        raise Exception(error_msg)
        
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

def process_single_user(username, user_id):
    """
    处理单个用户的人脸特征
    :param username: 用户名
    :param user_id: 用户ID
    :return: (success, message)
    """
    try:
        # 获取用户的人脸图片目录
        user_face_dir = os.path.join(path_images_from_camera, f'person_{username}')
        if not os.path.exists(user_face_dir):
            return False, f"用户 {username} 的人脸数据目录不存在"
            
        # 获取该目录下的所有图片
        image_paths = [os.path.join(user_face_dir, f) for f in os.listdir(user_face_dir)
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
                      
        if not image_paths:
            return False, f"用户 {username} 的目录中没有找到图片"
            
        # 提取所有图片的特征
        features_list = []
        for img_path in image_paths:
            try:
                # 读取图片
                img = cv2.imread(img_path)
                if img is None:
                    logging.warning(f"无法读取图片: {img_path}")
                    continue
                    
                # 检测人脸
                face_locations = face_recognition.face_locations(img)
                if not face_locations:
                    logging.warning(f"在图片中未检测到人脸: {img_path}")
                    continue
                    
                if len(face_locations) > 1:
                    logging.warning(f"在图片中检测到多个人脸: {img_path}")
                    continue
                    
                # 提取人脸特征
                face_encodings = face_recognition.face_encodings(img, face_locations)
                if not face_encodings:
                    logging.warning(f"无法提取人脸特征: {img_path}")
                    continue
                    
                features_list.append(face_encodings[0])
                
            except Exception as e:
                logging.error(f"处理图片时出错 {img_path}: {str(e)}")
                continue
                
        if not features_list:
            return False, "没有成功提取到任何人脸特征"
            
        # 计算平均特征
        features_mean = np.mean(features_list, axis=0)
        
        # 保存到数据库
        save_to_database(username, features_mean, user_id)
        
        return True, "人脸特征处理成功"
        
    except Exception as e:
        logging.error(f"处理用户 {username} 的人脸特征时出错: {str(e)}")
        return False, str(e)

def train_single_user(username):
    """
    训练指定用户的人脸数据
    :param username: 用户名
    :return: (bool, str) - (是否成功, 错误信息)
    """
    try:
        # 构建用户人脸文件夹路径
        user_face_dir = os.path.join(path_images_from_camera, f'person_{username}')
        
        # 检查文件夹是否存在
        if not os.path.exists(user_face_dir):
            return False, f"用户文件夹不存在：{user_face_dir}"
            
        logging.info(f"开始训练用户 {username} 的人脸数据")
        
        # 获取人脸特征
        features_mean = return_features_mean_personX(user_face_dir)
        
        if features_mean is None:
            return False, "无法提取有效的人脸特征"
            
        if isinstance(features_mean, np.ndarray):
            # 获取用户ID
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
            result = cursor.fetchone()
            if not result:
                return False, f"未找到用户 {username}"
            user_id = result[0]
            
            # 保存到数据库
            save_to_database(username, features_mean, user_id)
            logging.info(f"成功训练用户 {username} 的人脸特征")
            return True, "人脸特征训练成功"
        else:
            return False, "人脸特征数据格式错误"
            
    except Exception as e:
        error_msg = f"训练用户 {username} 的人脸数据时出错: {str(e)}"
        logging.error(error_msg)
        return False, error_msg
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals() and conn.is_connected():
            conn.close()

def main():
    """
    批量处理所有未处理的人脸数据
    """
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 获取数据库中已存在的用户名
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM known_faces")
        existing_names = set(row[0] for row in cursor.fetchall())
        cursor.close()
        conn.close()
        
        # 获取所有人脸文件夹
        person_list = os.listdir(path_images_from_camera)
        person_list.sort()
        
        processed_count = 0
        for person in person_list:
            # 检查文件夹名称格式
            if not person.startswith('person_'):
                logging.warning(f"跳过 {person}：文件夹名称格式不正确")
                continue
                
            # 提取用户名
            person_name = person[7:]  # 去掉 'person_' 前缀
            
            # 如果用户名已存在于数据库中，跳过处理
            if person_name in existing_names:
                logging.info(f"跳过 {person}：用户特征已存在于数据库中")
                continue
                
            # 处理新用户
            success, message = process_single_user(person_name, processed_count + 1)
            if success:
                processed_count += 1
            else:
                logging.error(message)
            
            logging.info('\n')
            
        if processed_count > 0:
            logging.info(f"成功处理 {processed_count} 个新用户的人脸特征")
        else:
            logging.info("没有新的人脸数据需要处理")
            
    except Exception as e:
        logging.error(f"处理过程中出错: {str(e)}")

if __name__ == '__main__':
    main()
import cv2
import face_recognition
import numpy as np
import logging
from db_config import get_db_connection

logger = logging.getLogger(__name__)

def face_recognition_test(image_path, known_faces):
    """
    测试人脸识别功能
    
    Args:
        image_path: 要识别的图像路径
        known_faces: 已知人脸特征列表
        
    Returns:
        dict: 包含识别结果的字典
    """
    try:
        # 读取图像
        frame = cv2.imread(image_path)
        if frame is None:
            return {
                'success': False,
                'message': '无法读取图像文件'
            }
        
        # 检测人脸
        face_locations = face_recognition.face_locations(frame)
        if not face_locations:
            return {
                'success': False,
                'message': '未检测到人脸'
            }
        
        if len(face_locations) > 1:
            return {
                'success': False,
                'message': '检测到多个人脸，请确保画面中只有一个人脸'
            }
        
        # 提取人脸特征
        face_encoding = face_recognition.face_encodings(frame, face_locations)[0]
        
        # 初始化最佳匹配
        best_match = None
        best_distance = float('inf')
        
        # 与已知人脸特征进行比对
        for face_data in known_faces:
            known_encoding = np.frombuffer(face_data['face_encoding'], dtype=np.float64)
            distance = np.linalg.norm(face_encoding - known_encoding)
            
            if distance < best_distance:
                best_distance = distance
                best_match = face_data['name']
        
        # 计算置信度
        confidence = (1 - best_distance) * 100
        
        # 如果置信度太低，认为是未知人脸
        if confidence < 60:
            return {
                'success': False,
                'message': '未能匹配到已知人脸',
                'confidence': confidence
            }
        
        return {
            'success': True,
            'name': best_match,
            'confidence': confidence
        }
        
    except Exception as e:
        return {
            'success': False,
            'message': f'处理过程出错：{str(e)}'
        }

def face_recognition_test_with_db(image_path, known_faces):
    """
    使用数据库中的人脸特征进行识别测试
    :param image_path: 图片路径
    :param known_faces: 数据库中的人脸特征列表，每项包含 id, name, face_encoding
    :return: 识别结果字典
    """
    try:
        # 加载图片
        image = cv2.imread(image_path)
        if image is None:
            return {
                'status': 'error',
                'message': '无法读取图像文件'
            }
            
        # 检测人脸位置
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            return {
                'status': 'error',
                'message': '未检测到人脸'
            }
            
        if len(face_locations) > 1:
            return {
                'status': 'error',
                'message': '检测到多个人脸，请确保画面中只有一个人脸'
            }
            
        # 提取人脸特征
        face_encodings = face_recognition.face_encodings(image, face_locations)
        if not face_encodings:
            return {
                'status': 'error',
                'message': '无法提取人脸特征'
            }
            
        face_encoding = face_encodings[0]
        
        # 初始化最佳匹配
        best_match = None
        best_distance = float('inf')
        best_user_id = None
        
        # 与数据库中的人脸特征进行比对
        for user_id, name, encoding in known_faces:
            try:
                # 将二进制数据转换为numpy数组
                known_encoding = np.frombuffer(encoding, dtype=np.float64)
                
                # 计算欧氏距离
                distance = np.linalg.norm(face_encoding - known_encoding)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = name
                    best_user_id = user_id
                    
            except Exception as e:
                logger.error(f"处理用户 {name} 的人脸特征时出错: {e}")
                continue
                
        # 计算置信度
        confidence = (1 - best_distance) * 100 if best_distance != float('inf') else 0
        
        # 判断是否匹配成功（阈值可调整）
        if best_distance <= 0.4 and best_match is not None:
            return {
                'status': 'success',
                'user_id': best_user_id,
                'name': best_match,
                'confidence': round(confidence, 2)
            }
        else:
            return {
                'status': 'error',
                'message': '未能匹配到已知人脸',
                'confidence': round(confidence, 2)
            }
            
    except Exception as e:
        logger.error(f"人脸识别过程出错: {e}")
        return {
            'status': 'error',
            'message': f'处理过程出错: {str(e)}'
        }

def get_next_available_id(cursor, table_name):
    """
    获取表中下一个可用的最小ID
    """
    cursor.execute(f"SELECT id FROM {table_name} ORDER BY id")
    used_ids = [row[0] for row in cursor.fetchall()]
    
    if not used_ids:
        return 1
        
    # 找出第一个可用的ID
    next_id = 1
    for current_id in used_ids:
        if current_id != next_id:
            return next_id
        next_id += 1
    return next_id

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
            logger.info(f"更新用户 {username} (ID: {user_id}) 的人脸特征")
        else:
            # 获取下一个可用的最小ID
            next_id = get_next_available_id(cursor, 'known_faces')
            
            # 插入新记录
            cursor.execute("""
                INSERT INTO known_faces (id, user_id, name, face_encoding, is_active)
                VALUES (%s, %s, %s, %s, TRUE)
            """, (next_id, user_id, username, face_encoding_binary))
            logger.info(f"插入用户 {username} (ID: {user_id}) 的人脸特征")
        
        # 提交事务
        conn.commit()
        logger.info(f"成功保存用户 {username} (ID: {user_id}) 的人脸特征到数据库")
        
    except ValueError as ve:
        logger.error(f"参数错误: {str(ve)}")
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        error_msg = f"保存人脸特征到数据库时出错: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)
        
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close() 
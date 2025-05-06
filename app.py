from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, session, flash, current_app
from flask_socketio import SocketIO, emit
import mysql.connector
from datetime import datetime, timedelta
import os
from attendance_taker import Face_Recognizer
from emotion_recognizer import EmotionRecognizer
import threading
import cv2
import numpy as np
import json
import time
import sqlite3
import dlib
import functools
import shutil
import base64
import logging
import face_recognition
from werkzeug.security import generate_password_hash, check_password_hash
from db_config import get_db_connection, logger
import calendar
import random
import torch
from models.database import Database
from models.face_recognition import FaceRecognizer
from models.attendance import Attendance
from utils.video_utils import generate_frames
from utils.face_utils import process_face_image, save_face_image
from utils.db_utils import get_user_by_id, get_user_by_username, insert_user, update_user_password, delete_user
from views.face_recognition_views import video_feed, check_in_page, check_out_page, attendance_query_page, attendance_statistics_page
from views.test_views import test_page
from controllers.face_recognition_controller import check_in, check_out, get_attendance_records, get_attendance_statistics
from controllers.test_controller import test_api
from blueprints.auth_bp import auth_bp
from blueprints.admin_bp import admin_bp
from blueprints.attendance_bp import attendance_bp

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建应用实例
app = Flask(
    __name__,
    template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates')),
    static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))
)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
socketio = SocketIO(app)

# 注册蓝图
app.register_blueprint(auth_bp)
app.register_blueprint(admin_bp)
app.register_blueprint(attendance_bp)

# 添加自定义过滤器
@app.template_filter('format_date')
def format_date(date):
    """格式化日期"""
    return date.strftime('%Y-%m-%d %H:%M:%S')

# 配置人脸数据集路径
app.config['FACE_DATASET_PATH'] = os.path.join(os.path.dirname(__file__), 'data/data_faces_from_camera')

# 配置临时文件目录
app.config['TEMP_FOLDER'] = os.path.join(os.path.dirname(__file__), 'temp')
# 确保临时目录存在
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

# MySQL配置
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',  # MySQL密码
    'database': 'attendance_db'
}

def get_db_connection():
    return mysql.connector.connect(**db_config)

# 配置GPU
def setup_gpu():
    """配置GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        logger.info('使用GPU加速')
        if dlib.DLIB_USE_CUDA:
            logger.info("dlib启用CUDA加速")
        else:
            logger.warning("dlib未编译CUDA支持")
        return device
    else:
        logger.warning("未检测到可用的GPU，将使用CPU")
        return torch.device('cpu')

# 在应用初始化时设置GPU
device = setup_gpu()

# 全局变量
face_recognizer = None
face_recognizer_lock = threading.Lock()

# 添加全局表情识别器
emotion_recognizer = None
emotion_recognizer_lock = threading.Lock()

def get_face_recognizer():
    global face_recognizer
    with face_recognizer_lock:
        if face_recognizer is None:
            face_recognizer = Face_Recognizer(device=device)
            
        try:
            face_recognizer.clear_known_faces()
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT kf.name, kf.face_encoding, kf.user_id 
                FROM known_faces kf 
                JOIN users u ON kf.user_id = u.id 
                WHERE kf.face_encoding IS NOT NULL
            """)
            known_faces = cursor.fetchall()
            
            logger.info(f"正在加载 {len(known_faces)} 个已知人脸数据")
            
            for name, encoding, user_id in known_faces:
                try:
                    face_encoding = np.frombuffer(encoding, dtype=np.float64)
                    if len(face_encoding) != 128:
                        logger.error(f"用户 {name} (ID: {user_id}) 的人脸数据长度错误: {len(face_encoding)}")
                        continue
                        
                    if np.any(np.isnan(face_encoding)) or np.all(face_encoding == 0):
                        logger.error(f"用户 {name} (ID: {user_id}) 的人脸数据无效")
                        continue
                        
                    face_recognizer.add_known_face(name, face_encoding)
                    logger.info(f"已加载用户 {name} (ID: {user_id}) 的人脸数据")
                except Exception as e:
                    logger.error(f"处理用户 {name} 的人脸数据时出错: {str(e)}")
                    continue
                    
            cursor.close()
            conn.close()
            
            total_faces = len(face_recognizer.known_face_encodings)
            logger.info(f"人脸数据加载完成，共加载 {total_faces} 个有效人脸特征")
            
            if total_faces == 0:
                logger.warning("警告：没有加载到任何有效的人脸数据！")
            
        except Exception as e:
            logger.error(f"加载已知人脸时出错: {str(e)}")
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
            
        return face_recognizer

def get_emotion_recognizer():
    global emotion_recognizer
    with emotion_recognizer_lock:
        if emotion_recognizer is None:
            emotion_recognizer = EmotionRecognizer()
        return emotion_recognizer

# 装饰器：要求用户登录
def login_required(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth.login'))
        return func(*args, **kwargs)
    return wrapper

# 装饰器：只允许管理员访问
def admin_required(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth.login'))
        if session.get('role') != 'admin':
            flash('需要管理员权限才能访问此页面', 'danger')
            return redirect(url_for('user_dashboard'))
        return func(*args, **kwargs)
    return wrapper

@app.route('/')
@login_required
def index():
    """首页根据角色跳转"""
    if session.get('role') == 'admin':
        return redirect(url_for('admin.dashboard')) 
    elif session.get('role') == 'user':
        return redirect(url_for('user_dashboard'))
    else:
        # 如果没有角色信息（理论上不应该发生），重定向到登录
        flash("无法确定用户角色，请重新登录", "warning")
        return redirect(url_for('auth.login'))

@app.route('/face_test')
@login_required
@admin_required
def face_test():
    """
    人脸识别测试页面
    """
    return render_template('face_test.html')

@app.route('/face_recognition')
@login_required
def face_recognition():
    return render_template('face_recognition.html')

@app.route('/init_face_recognizer', methods=['GET'])
@login_required
def init_face_recognizer():
    try:
        recognizer = get_face_recognizer()
        return jsonify({
            'success': True,
            'known_faces_count': len(recognizer.known_face_encodings)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

def generate_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("无法打开摄像头")
        return
        
    recognizer = get_face_recognizer()
    detector = dlib.get_frontal_face_detector()  # 使用dlib的人脸检测器
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
                
            # 添加水平翻转
            frame = cv2.flip(frame, 1)  # 1表示水平翻转
            
            # 检测人脸
            faces = detector(frame, 0)  # 使用dlib检测器
            
            # 在检测到的人脸周围画框
            for face in faces:
                left = face.left()
                top = face.top()
                right = face.right()
                bottom = face.bottom()
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
            # 将图像编码为JPEG格式
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.1)  # 控制帧率
            
    except Exception as e:
        print(f"视频流错误: {str(e)}")
    finally:
        camera.release()

@app.route('/video_feed')
def video_feed_route():
    return video_feed()

@app.route('/face_check_in', methods=['POST'])
@login_required
def face_check_in():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': '没有收到图像'})
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': '没有选择文件'})
        
    try:
        # 读取图像
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 使用人脸检测器
        detector = dlib.get_frontal_face_detector()
        faces = detector(image, 1)
        
        if not faces:
            return jsonify({'success': False, 'message': '未检测到人脸'})
            
        # 使用人脸识别器进行识别
        recognizer = get_face_recognizer()
        recognized_name = recognizer.recognize_face(image)
        
        # 进行表情识别
        face = faces[0]
        face_img = image[face.top():face.bottom(), face.left():face.right()]
        emotion_recognizer = get_emotion_recognizer()
        emotion_result = emotion_recognizer.predict_emotion(face_img)
        
        # 如果是普通用户，只能为自己打卡
        if session.get('role') == 'user':
            user_name = session.get('name')
            if recognized_name and recognized_name != user_name:
                return jsonify({
                    'success': False,
                    'message': f'识别到的人脸不是您 ({user_name})，无法打卡'
                })
            recognized_name = user_name  # 强制使用登录用户名
        
        if recognized_name:
            # 检查今日是否已打卡
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # 获取用户ID
            cursor.execute("SELECT id FROM users WHERE name = %s", (recognized_name,))
            user_result = cursor.fetchone()
            if not user_result:
                cursor.close()
                conn.close()
                return jsonify({
                    'success': False,
                    'message': f'未找到用户 {recognized_name}'
                })
            user_id = user_result[0]
            
            # 检查今日是否已打卡
            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute("""
                SELECT COUNT(*) FROM attendance 
                WHERE user_id = %s AND DATE(check_in_time) = %s
            """, (user_id, today))
            
            already_checked = cursor.fetchone()[0] > 0
            
            if already_checked:
                cursor.close()
                conn.close()
                return jsonify({
                    'success': False,
                    'message': f'{recognized_name} 今日已打卡'
                })
            
            # 记录考勤和表情
            current_time = datetime.now()
            cursor.execute("""
                INSERT INTO attendance (user_id, name, check_in_time, emotion) 
                VALUES (%s, %s, %s, %s)
            """, (user_id, recognized_name, current_time, 
                  emotion_result.get('emotion', '未知') if emotion_result['success'] else '未知'))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return jsonify({
                'success': True,
                'message': f'打卡成功！欢迎 {recognized_name}',
                'details': {
                    'emotion': emotion_result.get('emotion', '未知') if emotion_result['success'] else '未知',
                    'emotion_confidence': f"{emotion_result.get('confidence', 0)*100:.2f}%" if emotion_result['success'] else '0%'
                }
            })
        else:
            return jsonify({
                'success': False,
                'message': '未能识别到已知人脸'
            })
            
    except Exception as e:
        logger.error(f"签到失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'打卡失败：{str(e)}'
        })

@app.route('/face_recognition', methods=['POST'])
@login_required
def face_recognition_endpoint():
    try:
        with get_db_connection() as conn:
            with conn.cursor(dictionary=True) as cursor:
                # 获取上传的图片
                if 'image' not in request.files:
                    return jsonify({
                        'status': 'error',
                        'message': '未检测到图片'
                    }), 400
                
                image_file = request.files['image']
                if not image_file:
                    return jsonify({
                        'status': 'error',
                        'message': '图片上传失败'
                    }), 400
                
                # 保存临时文件
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{int(time.time())}.jpg')
                image_file.save(temp_path)
                
                try:
                    # 从数据库获取已知人脸特征
                    cursor.execute('''
                        SELECT u.id, u.name, kf.face_encoding 
                        FROM known_faces kf 
                        JOIN users u ON kf.user_id = u.id
                    ''')
                    known_faces = cursor.fetchall()
                    
                    if not known_faces:
                        return jsonify({
                            'status': 'error',
                            'message': '未找到已注册的人脸数据'
                        }), 404
                    
                    # 进行人脸识别
                    result = face_recognition_test_with_db(temp_path, known_faces)
                    
                    if result['status'] == 'success':
                        # 记录考勤
                        user_id = result['user_id']
                        now = datetime.now()
                        
                        cursor.execute('''
                            INSERT INTO attendance (user_id, check_in_time)
                            VALUES (%s, %s)
                        ''', (user_id, now))
                        conn.commit()
                        
                        return jsonify({
                            'status': 'success',
                            'message': f"识别成功：{result['name']}",
                            'confidence': result['confidence'],
                            'time': now.strftime('%H:%M:%S')
                        })
                    else:
                        return jsonify(result), 400
                        
                finally:
                    # 清理临时文件
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
    except Exception as e:
        logger.error(f"人脸识别过程中发生错误: {e}")
        return jsonify({
            'status': 'error',
            'message': '人脸识别失败，请稍后重试'
        }), 500

def get_attendance_records(user_name=None):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if user_name:
            cursor.execute(
                "SELECT name, check_in_time, DATE(check_in_time) as date FROM attendance WHERE name = %s ORDER BY check_in_time DESC",
                (user_name,)
            )
        else:
            cursor.execute(
                "SELECT name, check_in_time, DATE(check_in_time) as date FROM attendance ORDER BY check_in_time DESC"
            )
            
        records = cursor.fetchall()
        cursor.close()
        conn.close()
        return records
    except Exception as e:
        print(f"获取考勤记录失败: {str(e)}")
        return []

@app.route('/start_recognition', methods=['POST'])
@login_required
def start_recognition():
    camera = None
    try:
        # 尝试打开摄像头
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not camera.isOpened():
            logger.error("无法打开摄像头")
            return jsonify({
                "status": "error",
                "message": "无法打开摄像头，请检查摄像头是否正确连接"
            })

        # 读取图像
        success, frame = camera.read()
        if not success or frame is None:
            logger.error("无法读取摄像头画面")
            return jsonify({
                "status": "error",
                "message": "无法读取摄像头画面"
            })

        # 水平翻转图像（镜像）
        frame = cv2.flip(frame, 1)
        
        # 使用dlib的人脸检测器
        detector = dlib.get_frontal_face_detector()
        faces = detector(frame, 1)
        
        if len(faces) == 0:
            return jsonify({
                "status": "error",
                "message": "未检测到人脸"
            })
            
        if len(faces) > 1:
            return jsonify({
                "status": "error",
                "message": "检测到多个人脸，请确保画面中只有一个人脸"
            })
            
        # 获取人脸特征提取器
        shape_predictor = dlib.shape_predictor('data/data_dlib/data_dlib/shape_predictor_68_face_landmarks.dat')
        face_rec_model = dlib.face_recognition_model_v1('data/data_dlib/data_dlib/dlib_face_recognition_resnet_model_v1.dat')
        
        # 从数据库获取人脸特征和用户信息
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT u.id, u.name, k.face_encoding 
            FROM known_faces k
            JOIN users u ON k.user_id = u.id
            WHERE k.face_encoding IS NOT NULL
        """)
        known_faces = cursor.fetchall()
        
        if not known_faces:
            return jsonify({
                "status": "error",
                "message": "系统中还没有注册任何人脸数据"
            })
            
        # 获取第一个人脸的特征
        shape = shape_predictor(frame, faces[0])
        face_descriptor = face_rec_model.compute_face_descriptor(frame, shape)
        face_encoding = np.array(face_descriptor)
        
        # 初始化最佳匹配
        best_match = None
        best_distance = float('inf')
        best_user_id = None
        
        # 与数据库中的每个人脸特征进行比对
        for user_id, name, encoding_blob in known_faces:
            known_encoding = np.frombuffer(encoding_blob, dtype=np.float64)
            # 计算欧氏距离
            distance = np.linalg.norm(face_encoding - known_encoding)
            
            if distance < best_distance:
                best_distance = distance
                best_match = name
                best_user_id = user_id
                
        # 如果是普通用户，只能为自己打卡
        if session.get('role') == 'user':
            user_name = session.get('name')
            if best_match and best_match != user_name:
                return jsonify({
                    "status": "error",
                    "message": f"识别到的人脸不是您 ({user_name})，无法打卡"
                })
            best_match = user_name  # 强制使用登录用户名
            best_user_id = session.get('user_id')
                
        # 如果找到匹配的人脸，尝试打卡
        if best_distance <= 0.4:  # 使用0.4作为阈值
            # 检查是否已打卡
            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute(
                "SELECT COUNT(*) FROM attendance WHERE user_id = %s AND DATE(check_in_time) = %s",
                (best_user_id, today)
            )
            if cursor.fetchone()[0] > 0:
                return jsonify({
                    "status": "error",
                    "message": f"{best_match} 今日已打卡"
                })
                
            # 记录打卡
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute(
                "INSERT INTO attendance (user_id, name, check_in_time) VALUES (%s, %s, %s)",
                (best_user_id, best_match, current_time)
            )
            conn.commit()
            
            return jsonify({
                "status": "success",
                "message": f"打卡成功！欢迎 {best_match}",
                "details": {
                    "recognized_name": best_match,
                    "confidence": f"{(1 - best_distance) * 100:.2f}%"
                }
            })
        else:
            return jsonify({
                "status": "error",
                "message": "未能识别到已知人脸",
                "details": {
                    "recognized_name": "未知",
                    "confidence": f"{(1 - best_distance) * 100:.2f}%"
                }
            })
            
    except Exception as e:
        logger.error(f"人脸识别过程中出错：{str(e)}")
        return jsonify({
            "status": "error",
            "message": f"处理过程出错：{str(e)}"
        })
    finally:
        if camera is not None:
            camera.release()
            logger.info("已释放摄像头资源")
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

@app.route('/face_check_out')
@login_required
def face_check_out_page():
    return render_template('face_check_in.html')  # 使用相同的模板，因为界面相同

@app.route('/face_check_out', methods=['POST'])
@login_required
def face_check_out():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': '没有收到图像'})
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': '没有选择文件'})
        
    try:
        # 读取图像
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 使用人脸识别器进行识别
        recognizer = get_face_recognizer()
        recognized_name = recognizer.recognize_face(image)
        
        # 如果是普通用户，只能为自己签退
        if session.get('role') == 'user':
            user_name = session.get('name')
            if recognized_name and recognized_name != user_name:
                return jsonify({
                    'success': False,
                    'message': f'识别到的人脸不是您 ({user_name})，无法签退'
                })
            recognized_name = user_name  # 强制使用登录用户名
        
        if recognized_name:
            # 检查今日是否已签到且未签退
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # 获取用户ID
            cursor.execute("SELECT id FROM users WHERE name = %s", (recognized_name,))
            user_result = cursor.fetchone()
            if not user_result:
                cursor.close()
                conn.close()
                return jsonify({
                    'success': False,
                    'message': f'未找到用户 {recognized_name}'
                })
            user_id = user_result[0]
            
            # 查找今日签到但未签退的记录
            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute("""
                SELECT id FROM attendance 
                WHERE user_id = %s 
                AND DATE(check_in_time) = %s 
                AND check_out_time IS NULL
            """, (user_id, today))
            
            record = cursor.fetchone()
            if not record:
                cursor.close()
                conn.close()
                return jsonify({
                    'success': False,
                    'message': f'{recognized_name} 今日未签到或已签退'
                })
            
            # 更新签退时间
            current_time = datetime.now()
            cursor.execute("""
                UPDATE attendance 
                SET check_out_time = %s 
                WHERE id = %s
            """, (current_time, record[0]))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return jsonify({
                'success': True,
                'message': f'签退成功！再见 {recognized_name}'
            })
        else:
            return jsonify({
                'success': False,
                'message': '未能识别到已知人脸'
            })
            
    except Exception as e:
        logger.error(f"签退失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'签退失败：{str(e)}'
        })

@app.route('/face_recognition_test', methods=['POST'])
@login_required
def face_recognition_test_endpoint():
    """
    人脸识别测试接口 - 不写入数据库
    """
    if 'image' not in request.files:
        return jsonify({
            'status': 'error',
            'message': '未收到图像数据'
        })
    
    image_file = request.files['image']
    if not image_file:
        return jsonify({
            'status': 'error',
            'message': '图像数据无效'
        })
    
    try:
        # 读取图像
        file_bytes = image_file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise Exception("无法解析图像数据")
        
        # 水平翻转图像
        frame = cv2.flip(frame, 1)
        
        # 使用dlib的人脸检测器
        detector = dlib.get_frontal_face_detector()
        faces = detector(frame, 1)
        
        if len(faces) == 0:
            return jsonify({
                'status': 'error',
                'message': '未检测到人脸'
            })
        
        if len(faces) > 1:
            return jsonify({
                'status': 'warning',
                'message': '检测到多个人脸，将只处理第一个检测到的人脸'
            })
        
        # 获取人脸特征
        shape_predictor = dlib.shape_predictor('data/data_dlib/data_dlib/shape_predictor_68_face_landmarks.dat')
        face_rec_model = dlib.face_recognition_model_v1('data/data_dlib/data_dlib/dlib_face_recognition_resnet_model_v1.dat')
        
        shape = shape_predictor(frame, faces[0])
        face_descriptor = face_rec_model.compute_face_descriptor(frame, shape)
        face_encoding = np.array(face_descriptor)
        
        # 从数据库获取人脸特征
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name, face_encoding 
            FROM known_faces 
            WHERE face_encoding IS NOT NULL
        """)
        known_faces = cursor.fetchall()
        
        if not known_faces:
            return jsonify({
                'status': 'warning',
                'message': '系统中还没有注册任何人脸数据'
            })
        
        # 初始化最佳匹配
        best_match = None
        best_distance = float('inf')
        
        # 与数据库中的每个人脸特征进行比对
        for name, encoding_blob in known_faces:
            known_encoding = np.frombuffer(encoding_blob, dtype=np.float64)
            # 计算欧氏距离
            distance = np.linalg.norm(face_encoding - known_encoding)
            
            if distance < best_distance:
                best_distance = distance
                best_match = name
        
        # 计算置信度
        confidence = (1 - best_distance) * 100
        
        # 提取人脸区域进行表情识别
        face = faces[0]
        face_img = frame[face.top():face.bottom(), face.left():face.right()]
        
        # 进行表情识别
        emotion_recognizer = get_emotion_recognizer()
        emotion_result = emotion_recognizer.predict_emotion(face_img)
        
        # 根据置信度返回不同的结果
        if best_distance <= 0.4:  # 使用0.4作为阈值
            return jsonify({
                'status': 'success',
                'message': '识别成功',
                'details': {
                    'recognized_name': best_match,
                    'face_confidence': f"{confidence:.2f}%",
                    'emotion': emotion_result.get('emotion', '未知') if emotion_result['success'] else '未知',
                    'emotion_confidence': f"{emotion_result.get('confidence', 0)*100:.2f}%" if emotion_result['success'] else '0%'
                }
            })
        else:
            return jsonify({
                'status': 'error',
                'message': '未能匹配到已知人脸',
                'details': {
                    'recognized_name': '未知',
                    'face_confidence': f"{confidence:.2f}%",
                    'emotion': emotion_result.get('emotion', '未知') if emotion_result['success'] else '未知',
                    'emotion_confidence': f"{emotion_result.get('confidence', 0)*100:.2f}%" if emotion_result['success'] else '0%'
                }
            })
            
    except Exception as e:
        logger.error(f"人脸识别测试出错：{str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'处理过程出错：{str(e)}'
        })
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

@app.route('/face_test_api', methods=['POST'])
@login_required
def face_test_api():
    try:
        # 获取图像数据
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({'error': '未接收到图像数据'}), 400

        # 解码base64图像
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': '图像解码失败'}), 400

        # 获取人脸识别器实例
        face_recognizer = get_face_recognizer()
        emotion_recognizer = get_emotion_recognizer()

        # 进行人脸识别
        face_locations, face_names, face_distances = face_recognizer.recognize_faces(frame)
        
        if not face_locations:
            return jsonify({
                'status': 'error',
                'message': '未检测到人脸'
            }), 400

        # 获取第一个检测到的人脸
        face_location = face_locations[0]
        name = face_names[0]
        confidence = 1 - face_distances[0] if face_distances else 0

        # 进行表情识别
        emotion, emotion_confidence = emotion_recognizer.recognize_emotion(frame, face_location)

        return jsonify({
            'status': 'success',
            'name': name,
            'confidence': float(confidence),
            'emotion': emotion,
            'emotion_confidence': float(emotion_confidence)
        })

    except Exception as e:
        logger.error(f"人脸测试API错误: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'处理失败: {str(e)}'
        }), 500

@app.route('/performance_test', methods=['POST'])
@login_required
@admin_required
def performance_test():
    try:
        test_type = request.json.get('type')
        if not test_type:
            return jsonify({'status': 'error', 'message': '测试类型未指定'}), 400

        if test_type == 'recognition':
            # 创建测试图片目录（如果不存在）
            test_folder = os.path.join('data', 'test_images')
            os.makedirs(test_folder, exist_ok=True)
            
            # 首先尝试从测试文件夹获取图片
            image_files = [f for f in os.listdir(test_folder) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # 如果测试文件夹为空，则使用已注册用户的图片
            if not image_files:
                test_images = []
                for user_dir in os.listdir('data/data_faces_from_camera'):
                    user_path = os.path.join('data/data_faces_from_camera', user_dir)
                    if os.path.isdir(user_path):
                        for img in os.listdir(user_path):
                            if img.endswith(('.jpg', '.png', '.jpeg')):
                                test_images.append(os.path.join(user_path, img))
            else:
                # 使用测试文件夹中的图片
                test_images = [os.path.join(test_folder, f) for f in image_files]
            
            if not test_images:
                return jsonify({
                    'status': 'error', 
                    'message': '没有找到可用的测试图片。请确保 data/test_images 目录中有图片，或者已注册用户的人脸图片。'
                }), 404
            
            # 随机选择最多20张图片进行测试
            selected_images = random.sample(test_images, min(20, len(test_images)))
            
            total_time = 0
            success_count = 0
            recognized_count = 0
            recognition_results = []
            
            # 获取人脸识别器实例
            face_recognizer = get_face_recognizer()
            
            for image_path in selected_images:
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    start_time = time.time()
                    
                    # 执行人脸识别
                    result = face_recognizer.recognize_face(image)
                    
                    end_time = time.time()
                    process_time = (end_time - start_time) * 1000  # 转换为毫秒
                    
                    recognition_results.append({
                        'image_name': os.path.basename(image_path),
                        'recognized_name': result if result else '未识别',
                        'time': round(process_time, 2)
                    })
                    
                    if result:  # 如果成功识别出人脸
                        recognized_count += 1
                    total_time += process_time
                    success_count += 1
                    logger.info(f"识别成功: {os.path.basename(image_path)}, 耗时: {process_time:.2f}ms")
                    
                except Exception as e:
                    logger.error(f"处理图片时出错: {str(e)}")
                    continue
            
            if success_count == 0:
                return jsonify({
                    'status': 'error', 
                    'message': '未能成功处理任何图片'
                }), 400
                
            avg_time = total_time / success_count
            return jsonify({
                'status': 'success',
                'recognition_time': round(avg_time, 2),
                'tested_images': success_count,
                'recognized_count': recognized_count,
                'recognition_rate': round(recognized_count / success_count * 100, 2),
                'details': recognition_results
            })
            
        elif test_type == 'fps':
            # 测试视频流帧率
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return jsonify({
                    'status': 'error', 
                    'message': '无法打开摄像头'
                }), 500
                
            frame_count = 0
            recognized_count = 0
            start_time = time.time()
            duration = 5  # 测试5秒
            
            # 获取人脸识别器实例
            face_recognizer = get_face_recognizer()
            
            try:
                while time.time() - start_time < duration:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 执行人脸识别
                    result = face_recognizer.recognize_face(frame)
                    if result:
                        recognized_count += 1
                    
                    frame_count += 1
                    
                fps = frame_count / duration
                recognition_rate = recognized_count / frame_count * 100 if frame_count > 0 else 0
                
                return jsonify({
                    'status': 'success',
                    'fps': round(fps, 2),
                    'tested_frames': frame_count,
                    'recognized_frames': recognized_count,
                    'recognition_rate': round(recognition_rate, 2),
                    'duration': duration
                })
                
            finally:
                cap.release()
            
        elif test_type == 'page_load':
            # 测试页面加载时间
            total_time = 0
            test_count = 10  # 测试10次
            
            for _ in range(test_count):
                start_time = time.time()
                
                # 模拟页面加载
                with get_db_connection() as conn:
                    with conn.cursor() as cursor:
                        # 执行一些数据库查询
                        cursor.execute("SELECT COUNT(*) FROM users")
                        cursor.fetchone()
                        cursor.execute("SELECT COUNT(*) FROM attendance")
                        cursor.fetchone()
                        cursor.execute("SELECT COUNT(*) FROM known_faces")
                        cursor.fetchone()
                
                end_time = time.time()
                total_time += (end_time - start_time)
                
            avg_time = total_time / test_count
            return jsonify({
                'status': 'success',
                'load_time': round(avg_time, 2),
                'test_count': test_count
            })
            
        elif test_type == 'query':
            # 测试查询响应时间
            total_time = 0
            test_count = 10  # 测试10次
            
            for _ in range(test_count):
                start_time = time.time()
                
                # 执行考勤记录查询
                with get_db_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("""
                            SELECT a.id, u.name, a.check_in_time, a.check_out_time, a.emotion
                            FROM attendance a
                            JOIN users u ON a.user_id = u.id
                            ORDER BY a.check_in_time DESC
                            LIMIT 100
                        """)
                        cursor.fetchall()
                
                end_time = time.time()
                total_time += (end_time - start_time)
                
            avg_time = total_time / test_count
            return jsonify({
                'status': 'success',
                'query_time': round(avg_time, 2),
                'test_count': test_count
            })
            
        else:
            return jsonify({
                'status': 'error', 
                'message': '无效的测试类型'
            }), 400
            
    except Exception as e:
        logger.error(f"性能测试失败: {str(e)}")
        return jsonify({
            'status': 'error', 
            'message': f'测试失败: {str(e)}'
        }), 500

@app.route('/test_with_random_image', methods=['POST'])
@login_required
@admin_required
def test_with_random_image():
    try:
        # 获取测试文件夹路径
        test_folder = request.form.get('test_folder', 'data/test_images')
        if not os.path.exists(test_folder):
            return jsonify({
                'status': 'error',
                'message': f'测试文件夹 {test_folder} 不存在'
            })
            
        # 获取所有图片文件
        image_files = [f for f in os.listdir(test_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                      
        if not image_files:
            return jsonify({
                'status': 'error',
                'message': '测试文件夹中没有找到图片文件'
            })
            
        # 随机选择一张图片
        random_image = random.choice(image_files)
        image_path = os.path.join(test_folder, random_image)
        
        # 读取图片
        frame = cv2.imread(image_path)
        if frame is None:
            return jsonify({
                'status': 'error',
                'message': f'无法读取图片: {random_image}'
            })
            
        # 获取人脸识别器
        recognizer = get_face_recognizer()
        
        # 进行人脸识别
        recognized_name = recognizer.recognize_face(frame)
        
        if recognized_name:
            return jsonify({
                'status': 'success',
                'message': f'识别成功',
                'details': {
                    'image_name': random_image,
                    'recognized_name': recognized_name
                }
            })
        else:
            return jsonify({
                'status': 'error',
                'message': '未能识别到已知人脸',
                'details': {
                    'image_name': random_image
                }
            })
            
    except Exception as e:
        logger.error(f"随机图片测试出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'测试过程出错: {str(e)}'
        })

@app.route('/check_in_page')
def check_in_page_route():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    return render_template('face_check_in.html', mode='check_in')

@app.route('/check_out_page')
def check_out_page_route():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    return render_template('face_check_in.html', mode='check_out')

@app.route('/attendance_query_page')
def attendance_query_page_route():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    return attendance_query_page()

@app.route('/attendance_statistics_page')
def attendance_statistics_page_route():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    return attendance_statistics_page()

@app.route('/test_page')
def test_page_route():
    return test_page()

@app.route('/check_in', methods=['POST'])
def check_in_route():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})
    return check_in()

@app.route('/check_out', methods=['POST'])
def check_out_route():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})
    return check_out()

@app.route('/get_attendance_records')
def get_attendance_records_route():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})
    return get_attendance_records()

@app.route('/get_attendance_statistics')
def get_attendance_statistics_route():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})
    return get_attendance_statistics()

@app.route('/test_api', methods=['POST'])
def test_api_route():
    return test_api()

def face_recognition_test_with_db(image_path=None, known_faces=None):
    try:
        # 获取人脸识别器
        recognizer = get_face_recognizer()
        
        # 如果提供了图像路径，则读取图像
        if image_path:
            frame = cv2.imread(image_path)
            if frame is None:
                return {'status': 'error', 'message': '无法读取图像'}
        else:
            # 否则使用摄像头
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                return {'status': 'error', 'message': '无法打开摄像头'}
            ret, frame = camera.read()
            camera.release()
            if not ret:
                return {'status': 'error', 'message': '无法读取摄像头画面'}
        
        # 水平翻转图像
        frame = cv2.flip(frame, 1)
        
        # 使用dlib的人脸检测器
        detector = dlib.get_frontal_face_detector()
        faces = detector(frame, 1)
        
        if len(faces) == 0:
            return {'status': 'error', 'message': '未检测到人脸'}
        
        if len(faces) > 1:
            return {'status': 'error', 'message': '检测到多个人脸，请确保画面中只有一个人脸'}
        
        # 获取人脸特征
        shape_predictor = dlib.shape_predictor('data/data_dlib/data_dlib/shape_predictor_68_face_landmarks.dat')
        face_rec_model = dlib.face_recognition_model_v1('data/data_dlib/data_dlib/dlib_face_recognition_resnet_model_v1.dat')
        
        # 如果提供了已知人脸数据，则使用它
        if known_faces:
            known_face_data = known_faces
        else:
            # 否则从数据库获取
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT u.id, u.name, k.face_encoding 
                FROM known_faces k
                JOIN users u ON k.user_id = u.id
                WHERE k.face_encoding IS NOT NULL
            """)
            known_face_data = cursor.fetchall()
            cursor.close()
            conn.close()
        
        if not known_face_data:
            return {'status': 'error', 'message': '系统中还没有注册任何人脸数据'}
        
        # 获取第一个人脸的特征
        shape = shape_predictor(frame, faces[0])
        face_descriptor = face_rec_model.compute_face_descriptor(frame, shape)
        face_encoding = np.array(face_descriptor)
        
        # 初始化最佳匹配
        best_match = None
        best_distance = float('inf')
        best_user_id = None
        
        # 与数据库中的每个人脸特征进行比对
        for user_id, name, encoding_blob in known_face_data:
            known_encoding = np.frombuffer(encoding_blob, dtype=np.float64)
            # 计算欧氏距离
            distance = np.linalg.norm(face_encoding - known_encoding)
            
            if distance < best_distance:
                best_distance = distance
                best_match = name
                best_user_id = user_id
        
        # 计算置信度
        confidence = (1 - best_distance) * 100
        
        # 提取人脸区域进行表情识别
        face = faces[0]
        face_img = frame[face.top():face.bottom(), face.left():face.right()]
        
        # 进行表情识别
        emotion_recognizer = get_emotion_recognizer()
        emotion_result = emotion_recognizer.predict_emotion(face_img)
        
        # 根据置信度返回不同的结果
        if best_distance <= 0.4:  # 使用0.4作为阈值
            return {
                'status': 'success',
                'message': '识别成功',
                'details': {
                    'recognized_name': best_match,
                    'user_id': best_user_id,
                    'face_confidence': f"{confidence:.2f}%",
                    'emotion': emotion_result.get('emotion', '未知') if emotion_result['success'] else '未知',
                    'emotion_confidence': f"{emotion_result.get('confidence', 0)*100:.2f}%" if emotion_result['success'] else '0%'
                }
            }
        else:
            return {
                'status': 'error',
                'message': '未能匹配到已知人脸',
                'details': {
                    'recognized_name': '未知',
                    'face_confidence': f"{confidence:.2f}%",
                    'emotion': emotion_result.get('emotion', '未知') if emotion_result['success'] else '未知',
                    'emotion_confidence': f"{emotion_result.get('confidence', 0)*100:.2f}%" if emotion_result['success'] else '0%'
                }
            }
            
    except Exception as e:
        logger.error(f"人脸识别测试出错：{str(e)}")
        return {'status': 'error', 'message': f'处理过程出错：{str(e)}'}

@app.route('/user_dashboard')
@login_required
def user_dashboard():
    """用户仪表盘"""
    # 这里可以添加获取用户特定数据的逻辑
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('auth.login'))
        
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    
    # 获取最近打卡记录
    cursor.execute("""
        SELECT DATE(check_in_time) as check_date, TIME_FORMAT(check_in_time, '%H:%i:%s') as check_time
        FROM attendance
        WHERE user_id = %s
        ORDER BY check_in_time DESC
        LIMIT 5
    """, (user_id,))
    recent_records = cursor.fetchall()
    
    # 获取本月打卡天数
    today = datetime.now()
    start_of_month = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    cursor.execute("""
        SELECT COUNT(DISTINCT DATE(check_in_time))
        FROM attendance
        WHERE user_id = %s AND check_in_time >= %s
    """, (user_id, start_of_month))
    monthly_days = cursor.fetchone()['COUNT(DISTINCT DATE(check_in_time))']
    
    # 检查今天是否已打卡
    today_str = today.strftime('%Y-%m-%d')
    cursor.execute("SELECT COUNT(*) FROM attendance WHERE user_id = %s AND DATE(check_in_time) = %s", (user_id, today_str))
    checked_today = cursor.fetchone()['COUNT(*)'] > 0
    
    cursor.close()
    conn.close()
    
    # 模拟已工作天数，实际应从数据库获取
    user['days_worked'] = 100 # 示例数据
    
    return render_template('user-dashboard.html', 
                             user=user,
                             recent_records=recent_records,
                             monthly_days=monthly_days,
                             checked_today=checked_today,
                             now_str=today.strftime('%Y年%m月%d日') # 添加当前日期字符串
                             )

if __name__ == '__main__':
    # 确保上传文件夹存在
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
        
    print("\n请在浏览器中访问：http://127.0.0.1:5000 或 http://localhost:5000\n")
    # 运行应用
    socketio.run(app, debug=True, host='127.0.0.1', port=5000)
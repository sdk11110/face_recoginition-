from flask import Blueprint, render_template, request, jsonify, Response, redirect, url_for, session, flash, current_app
import mysql.connector
from datetime import datetime, timedelta
import os
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
import torch
import random

from attendance_taker import Face_Recognizer
from emotion_recognizer import EmotionRecognizer
from face_features_processor import main as process_face_features, process_single_user, train_single_user
from face_recognition_processor import face_recognition_test, face_recognition_test_with_db
from db_config import get_db_connection, logger
from models.database import db
from models.face_recognition import face_recognizer
from werkzeug.utils import secure_filename

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建蓝图
admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

# 全局变量
face_recognizer = None
face_recognizer_lock = threading.Lock()
emotion_recognizer = None
emotion_recognizer_lock = threading.Lock()

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

# 配置GPU
def setup_gpu():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        logger.info(f"使用GPU加速: {torch.cuda.get_device_name(0)}")
        if dlib.DLIB_USE_CUDA:
            logger.info("dlib启用CUDA加速")
        else:
            logger.warning("dlib未编译CUDA支持")
        return device
    else:
        logger.warning("未检测到可用的GPU，将使用CPU")
        return torch.device('cpu')

# 在蓝图初始化时设置GPU
device = setup_gpu()

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

@admin_bp.route('/dashboard')
@admin_required
def dashboard():
    """管理员仪表盘"""
    try:
        if not db.connect():
            flash('数据库连接失败', 'danger')
            # 提供默认值以避免模板渲染错误
            now = datetime.now()
            current_date = f"{now.year}年{now.month}月{now.day}日"
            return render_template('admin-dashboard.html',
                                   stats={'total_users': 0, 'admin_count': 0, 'user_count': 0},
                                   recent_attendance=[],
                                   today_attendance=0,
                                   user_count=0,
                                   attendance_rate=0.0,
                                   weekly_attendance=[0]*7,
                                   current_date=current_date
                                   )

        # 获取用户统计
        db.cursor.execute("""
            SELECT
                COUNT(*) as total_users,
                SUM(CASE WHEN role = 'admin' THEN 1 ELSE 0 END) as admin_count,
                SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END) as user_count
            FROM users
            WHERE is_active = TRUE
        """)
        stats_result = db.cursor.fetchone()
        stats_dict = {'total_users': stats_result[0] if stats_result else 0,
                      'admin_count': stats_result[1] if stats_result else 0,
                      'user_count': stats_result[2] if stats_result else 0}


        # 获取最近的考勤记录 (只获取需要的字段)
        db.cursor.execute("""
            SELECT u.name as user_name, a.check_in_time
            FROM attendance a
            JOIN users u ON a.user_id = u.id
            ORDER BY a.check_in_time DESC
            LIMIT 5
        """)
        recent_attendance = db.cursor.fetchall()

        # --- 计算仪表盘所需数据 ---
        today_str = datetime.now().strftime('%Y-%m-%d')
        db.cursor.execute("""
            SELECT COUNT(DISTINCT user_id) FROM attendance WHERE DATE(check_in_time) = %s
        """, (today_str,))
        today_attendance_result = db.cursor.fetchone()
        today_attendance = today_attendance_result[0] if today_attendance_result else 0

        user_count = stats_dict['total_users']

        weekly_attendance = [0] * 7
        today = datetime.now().date()
        # Monday as start of week (weekday() == 0)
        start_of_week = today - timedelta(days=today.weekday())

        for i in range(7):
            current_day = start_of_week + timedelta(days=i)
            current_day_str = current_day.strftime('%Y-%m-%d')
            db.cursor.execute("""
                SELECT COUNT(DISTINCT user_id) FROM attendance WHERE DATE(check_in_time) = %s
            """, (current_day_str,))
            day_count_result = db.cursor.fetchone()
            weekly_attendance[i] = day_count_result[0] if day_count_result else 0

        # 示例出勤率计算 (今日出勤/总用户数)
        attendance_rate = round((today_attendance / user_count * 100), 1) if user_count > 0 else 0.0
        # --- 结束计算 ---

        db.close()

        # 传递所有计算好的变量到模板
        now = datetime.now()
        current_date = f"{now.year}年{now.month}月{now.day}日"
        return render_template('admin-dashboard.html',
                             stats=stats_dict, # 传递字典
                             recent_attendance=recent_attendance,
                             today_attendance=today_attendance,
                             user_count=user_count,
                             attendance_rate=attendance_rate,
                             weekly_attendance=weekly_attendance,
                             current_date=current_date
                             )

    except Exception as e:
        logger.error(f"获取仪表盘数据失败: {str(e)}")
        flash('获取数据失败，请稍后重试', 'danger')
        # 同样提供默认值
        now = datetime.now()
        current_date = f"{now.year}年{now.month}月{now.day}日"
        return render_template('admin-dashboard.html',
                               stats={'total_users': 0, 'admin_count': 0, 'user_count': 0},
                               recent_attendance=[],
                               today_attendance=0,
                               user_count=0,
                               attendance_rate=0.0,
                               weekly_attendance=[0]*7,
                               current_date=current_date
                               )

@admin_bp.route('/users')
@admin_required
def users():
    """用户管理页面"""
    try:
        if not db.connect():
            flash('数据库连接失败', 'danger')
            return render_template('admin/users.html')
            
        db.cursor.execute("""
            SELECT id, username, name, role, created_at, is_active
            FROM users
            ORDER BY created_at DESC
        """)
        users = db.cursor.fetchall()
        
        db.close()
        
        return render_template('admin/users.html', users=users)
        
    except Exception as e:
        logger.error(f"获取用户列表失败: {str(e)}")
        flash('获取用户列表失败，请稍后重试', 'danger')
        return render_template('admin/users.html')

@admin_bp.route('/user/<int:user_id>/toggle', methods=['POST'])
@admin_required
def toggle_user(user_id):
    """切换用户状态"""
    try:
        if not db.connect():
            return jsonify({'status': 'error', 'message': '数据库连接失败'}), 500
            
        db.cursor.execute("""
            UPDATE users 
            SET is_active = NOT is_active 
            WHERE id = %s AND role != 'admin'
        """, (user_id,))
        
        if db.cursor.rowcount == 0:
            db.close()
            return jsonify({'status': 'error', 'message': '用户不存在或无法修改管理员状态'}), 400
            
        db.conn.commit()
        db.close()
        
        return jsonify({'status': 'success', 'message': '用户状态已更新'})
        
    except Exception as e:
        logger.error(f"切换用户状态失败: {str(e)}")
        return jsonify({'status': 'error', 'message': '服务器内部错误'}), 500

@admin_bp.route('/face_data')
@admin_required
def face_data():
    """人脸数据管理页面"""
    try:
        if not db.connect():
            flash('数据库连接失败', 'danger')
            return render_template('admin/face_data.html')
            
        db.cursor.execute("""
            SELECT kf.*, u.name as user_name
            FROM known_faces kf
            JOIN users u ON kf.user_id = u.id
            ORDER BY kf.created_at DESC
        """)
        face_data = db.cursor.fetchall()
        
        db.close()
        
        return render_template('admin/face_data.html', face_data=face_data)
        
    except Exception as e:
        logger.error(f"获取人脸数据失败: {str(e)}")
        flash('获取人脸数据失败，请稍后重试', 'danger')
        return render_template('admin/face_data.html')

@admin_bp.route('/add_face', methods=['POST'])
@admin_required
def add_face():
    """添加人脸数据"""
    try:
        user_id = request.form.get('user_id')
        image_data = request.form.get('image')
        
        if not user_id or not image_data:
            return jsonify({'status': 'error', 'message': '缺少必要参数'}), 400
            
        # 解码图片
        try:
            image_data = base64.b64decode(image_data.split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"图片解码失败: {str(e)}")
            return jsonify({'status': 'error', 'message': '图片格式错误'}), 400
            
        # 添加人脸数据
        success = face_recognizer.add_face(user_id, image)
        
        if not success:
            return jsonify({'status': 'error', 'message': '添加人脸数据失败'}), 400
            
        return jsonify({'status': 'success', 'message': '人脸数据添加成功'})
        
    except Exception as e:
        logger.error(f"添加人脸数据失败: {str(e)}")
        return jsonify({'status': 'error', 'message': '服务器内部错误'}), 500

@admin_bp.route('/delete_face/<int:face_id>', methods=['POST'])
@admin_required
def delete_face(face_id):
    """删除人脸数据"""
    try:
        if not db.connect():
            return jsonify({'status': 'error', 'message': '数据库连接失败'}), 500
            
        db.cursor.execute("DELETE FROM known_faces WHERE id = %s", (face_id,))
        
        if db.cursor.rowcount == 0:
            db.close()
            return jsonify({'status': 'error', 'message': '人脸数据不存在'}), 400
            
        db.conn.commit()
        db.close()
        
        return jsonify({'status': 'success', 'message': '人脸数据删除成功'})
        
    except Exception as e:
        logger.error(f"删除人脸数据失败: {str(e)}")
        return jsonify({'status': 'error', 'message': '服务器内部错误'}), 500

@admin_bp.route('/face_test')
@admin_required
def face_test():
    return render_template('face_test.html')

@admin_bp.route('/face_management')
@admin_required
def face_management():
    try:
        with get_db_connection() as conn: # 使用 with 语句
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute("""
                SELECT u.id, u.name, u.username, COUNT(kf.id) as face_count
                FROM users u
                LEFT JOIN known_faces kf ON u.id = kf.user_id
                GROUP BY u.id, u.name, u.username
            """)
            users = cursor.fetchall()
            
            # with语句会自动关闭连接和游标，无需手动关闭
            # cursor.close()
            # conn.close()
        
        return render_template('face_management.html', users=users)
    except Exception as e:
        logger.error(f"获取用户管理人脸数据失败: {str(e)}")
        flash('加载用户管理人脸数据失败，请稍后重试。', 'danger')
        # 可以重定向回仪表盘或显示错误页面
        return redirect(url_for('admin.dashboard'))

@admin_bp.route('/add_user', methods=['POST'])
@admin_required
def add_user():
    try:
        data = request.get_json()
        name = data.get('name')
        username = data.get('username')
        password = data.get('password')
        role = data.get('role', 'user')
        
        if not all([name, username, password]):
            return jsonify({'success': False, 'message': '请填写所有必填字段'})
            
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 检查用户名是否已存在
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': '用户名已存在'})
            
        # 添加新用户
        cursor.execute("""
            INSERT INTO users (name, username, password, role)
            VALUES (%s, %s, %s, %s)
        """, (name, username, password, role))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': '用户添加成功'})
        
    except Exception as e:
        logger.error(f"添加用户时出错: {str(e)}")
        return jsonify({'success': False, 'message': f'添加用户失败: {str(e)}'})

@admin_bp.route('/delete_user/<int:user_id>', methods=['POST'])
@admin_required
def delete_user(user_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 删除用户的人脸数据
        cursor.execute("DELETE FROM known_faces WHERE user_id = %s", (user_id,))
        
        # 删除用户的考勤记录
        cursor.execute("DELETE FROM attendance WHERE user_id = %s", (user_id,))
        
        # 删除用户
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': '用户删除成功'})
        
    except Exception as e:
        logger.error(f"删除用户时出错: {str(e)}")
        return jsonify({'success': False, 'message': f'删除用户失败: {str(e)}'})

@admin_bp.route('/face_register')
@admin_required
def face_register():
    return render_template('face_register.html')

@admin_bp.route('/register_face', methods=['POST'])
@admin_required
def register_face():
    try:
        if 'face_image' not in request.files:
            return jsonify({'success': False, 'message': '未收到人脸图像'})
            
        face_image = request.files['face_image']
        user_id = request.form.get('user_id')
        
        if not user_id:
            return jsonify({'success': False, 'message': '未指定用户ID'})
            
        # 保存上传的图像
        temp_path = os.path.join(current_app.config['TEMP_FOLDER'], f'temp_{user_id}.jpg')
        face_image.save(temp_path)
        
        # 处理人脸特征
        face_encoding = process_single_user(temp_path)
        
        if face_encoding is None:
            os.remove(temp_path)
            return jsonify({'success': False, 'message': '无法检测到人脸或人脸特征提取失败'})
            
        # 保存到数据库
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO known_faces (user_id, face_encoding)
            VALUES (%s, %s)
        """, (user_id, face_encoding.tobytes()))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # 清理临时文件
        os.remove(temp_path)
        
        return jsonify({'success': True, 'message': '人脸注册成功'})
        
    except Exception as e:
        logger.error(f"注册人脸时出错: {str(e)}")
        return jsonify({'success': False, 'message': f'注册人脸失败: {str(e)}'})

@admin_bp.route('/train_face', methods=['POST'])
@admin_required
def train_face():
    try:
        process_face_features()
        return jsonify({'success': True, 'message': '人脸训练完成'})
    except Exception as e:
        logger.error(f"训练人脸时出错: {str(e)}")
        return jsonify({'success': False, 'message': f'训练人脸失败: {str(e)}'})

@admin_bp.route('/test_face_recognition', methods=['POST'])
@admin_required
def test_face_recognition():
    try:
        if 'test_image' not in request.files:
            return jsonify({'success': False, 'message': '未收到测试图像'})
            
        test_image = request.files['test_image']
        temp_path = os.path.join(current_app.config['TEMP_FOLDER'], 'test_image.jpg')
        test_image.save(temp_path)
        
        result = face_recognition_test(temp_path)
        
        os.remove(temp_path)
        
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        logger.error(f"测试人脸识别时出错: {str(e)}")
        return jsonify({'success': False, 'message': f'测试人脸识别失败: {str(e)}'})

@admin_bp.route('/performance_test', methods=['POST'])
@admin_required
def performance_test():
    try:
        test_count = int(request.form.get('test_count', 10))
        results = []
        
        for i in range(test_count):
            start_time = time.time()
            result = face_recognition_test_with_db()
            end_time = time.time()
            
            results.append({
                'test_id': i + 1,
                'recognition_time': end_time - start_time,
                'result': result
            })
            
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        logger.error(f"性能测试时出错: {str(e)}")
        return jsonify({'success': False, 'message': f'性能测试失败: {str(e)}'})

@admin_bp.route('/test_with_random_image', methods=['POST'])
@admin_required
def test_with_random_image():
    try:
        if 'test_image' not in request.files:
            return jsonify({'success': False, 'message': '未收到测试图像'})
            
        test_image = request.files['test_image']
        temp_path = os.path.join(current_app.config['TEMP_FOLDER'], 'random_test.jpg')
        test_image.save(temp_path)
        
        # 随机选择数据库中的一张人脸进行测试
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM known_faces ORDER BY RAND() LIMIT 1")
        random_face = cursor.fetchone()
        
        if not random_face:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': '数据库中没有可用的人脸数据'})
            
        result = face_recognition_test_with_db(temp_path, random_face[0])
        
        cursor.close()
        conn.close()
        os.remove(temp_path)
        
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        logger.error(f"随机图像测试时出错: {str(e)}")
        return jsonify({'success': False, 'message': f'随机图像测试失败: {str(e)}'}) 
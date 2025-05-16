from flask import Blueprint, render_template, request, jsonify, redirect, url_for, session, flash
from datetime import datetime, time
import os
import cv2
import numpy as np
import dlib
import functools
import logging
from db_config import get_db_connection, logger
from attendance_taker import Face_Recognizer
from emotion_recognizer import EmotionRecognizer
from models.database import db
from models.face_recognition import face_recognizer
from models.attendance import attendance
import base64

# 创建蓝图
attendance_bp = Blueprint('attendance', __name__, url_prefix='/attendance')

# 装饰器：要求用户登录
def login_required(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth.login'))
        return func(*args, **kwargs)
    return wrapper

@attendance_bp.route('/query')
@login_required
def attendance_query():
    try:
        # 获取查询参数
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # 修复：使用 with 语句正确使用上下文管理器
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            
            # 构建查询条件
            query_conditions = []
            query_params = []
            
            if start_date:
                query_conditions.append("DATE(a.check_in_time) >= %s")
                query_params.append(start_date)
            if end_date:
                query_conditions.append("DATE(a.check_in_time) <= %s")
                query_params.append(end_date)
                
            # 根据用户角色决定查询范围
            if session.get('role') != 'admin':
                query_conditions.append("u.id = %s")
                query_params.append(session.get('user_id'))
            
            # 构建完整的SQL查询
            base_query = """
                SELECT 
                    u.name,
                    DATE(a.check_in_time) as date,
                    TIME(a.check_in_time) as check_in,
                    TIME(a.check_out_time) as check_out,
                    CASE 
                        WHEN TIME(a.check_in_time) > '09:00:00' THEN '迟到'
                        ELSE '正常'
                    END as status,
                    CASE 
                        WHEN a.check_out_time IS NULL THEN '未签退'
                        WHEN TIME(a.check_out_time) < '18:00:00' THEN '早退'
                        ELSE '正常'
                    END as leave_status
                FROM attendance a
                JOIN users u ON a.user_id = u.id
            """
            
            if query_conditions:
                base_query += " WHERE " + " AND ".join(query_conditions)
            
            base_query += " ORDER BY a.check_in_time DESC"
            
            # 执行查询
            cursor.execute(base_query, query_params)
            attendance_data = cursor.fetchall()
            
            # 不需要手动关闭cursor和conn，上下文管理器会处理
        
        return render_template(
            'attendance_query.html',
            attendance_data=attendance_data,
            no_data=len(attendance_data) == 0,
            start_date=start_date,
            end_date=end_date,
            is_admin=session.get('role') == 'admin'
        )
        
    except Exception as e:
        logger.error(f"查询考勤记录失败: {str(e)}")
        flash(f'获取数据失败: {str(e)}', 'danger')
        
        # 确保错误页面传递了必要的参数，避免模板渲染错误
        # 创建一个返回空内容的默认值
        return render_template('attendance_query.html', 
                              attendance_data=[],
                              no_data=True,
                              start_date=None,
                              end_date=None,
                              is_admin=session.get('role') == 'admin',
                              error_message=str(e))

@attendance_bp.route('/check_in_page')
@login_required
def face_check_in_page():
    return render_template('face_check_in.html', mode='check_in')

@attendance_bp.route('/check_in', methods=['GET', 'POST'])
@login_required
def check_in():
    """打卡页面"""
    if request.method == 'POST':
        try:
            image_data = request.form.get('image')
            
            if not image_data:
                return jsonify({'status': 'error', 'message': '缺少图片数据'}), 400
                
            # 解码图片
            try:
                image_data = base64.b64decode(image_data.split(',')[1])
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                logger.error(f"图片解码失败: {str(e)}")
                return jsonify({'status': 'error', 'message': '图片格式错误'}), 400
                
            # 识别人脸
            face_locations, face_encodings = face_recognizer.detect_faces(image)
            
            if not face_locations:
                return jsonify({'status': 'error', 'message': '未检测到人脸'}), 400
                
            # 识别用户
            names = face_recognizer.recognize_faces(face_encodings)
            
            if not names or names[0] != session['name']:
                return jsonify({'status': 'error', 'message': '人脸识别失败'}), 400
                
            # 记录打卡
            current_time = datetime.now()
            is_late = current_time.time() > time(9, 0)  # 9点后算迟到
            
            success = attendance.record_check_in(session['user_id'], current_time, is_late)
            
            if not success:
                return jsonify({'status': 'error', 'message': '打卡失败'}), 400
                
            return jsonify({
                'status': 'success',
                'message': '打卡成功',
                'is_late': is_late
            })
            
        except Exception as e:
            logger.error(f"打卡失败: {str(e)}")
            return jsonify({'status': 'error', 'message': '服务器内部错误'}), 500
            
    return render_template('attendance/check_in.html')

@attendance_bp.route('/check_out_page')
@login_required
def face_check_out_page():
    return render_template('face_check_in.html', mode='check_out')

@attendance_bp.route('/check_out', methods=['GET', 'POST'])
@login_required
def check_out():
    """签退页面"""
    if request.method == 'POST':
        try:
            image_data = request.form.get('image')
            
            if not image_data:
                return jsonify({'status': 'error', 'message': '缺少图片数据'}), 400
                
            # 解码图片
            try:
                image_data = base64.b64decode(image_data.split(',')[1])
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                logger.error(f"图片解码失败: {str(e)}")
                return jsonify({'status': 'error', 'message': '图片格式错误'}), 400
                
            # 识别人脸
            face_locations, face_encodings = face_recognizer.detect_faces(image)
            
            if not face_locations:
                return jsonify({'status': 'error', 'message': '未检测到人脸'}), 400
                
            # 识别用户
            names = face_recognizer.recognize_faces(face_encodings)
            
            if not names or names[0] != session['name']:
                return jsonify({'status': 'error', 'message': '人脸识别失败'}), 400
                
            # 记录签退
            current_time = datetime.now()
            is_early = current_time.time() < time(18, 0)  # 18点前算早退
            
            success = attendance.record_check_out(session['user_id'], current_time, is_early)
            
            if not success:
                return jsonify({'status': 'error', 'message': '签退失败'}), 400
                
            return jsonify({
                'status': 'success',
                'message': '签退成功',
                'is_early': is_early
            })
            
        except Exception as e:
            logger.error(f"签退失败: {str(e)}")
            return jsonify({'status': 'error', 'message': '服务器内部错误'}), 500
            
    return render_template('attendance/check_out.html')

@attendance_bp.route('/statistics')
@login_required
def attendance_statistics():
    try:
        # 获取用户选择的月份，默认为当前月份
        selected_month_str = request.args.get('month')
        if selected_month_str:
            try:
                selected_date = datetime.strptime(selected_month_str, '%Y-%m')
                year = selected_date.year
                month = selected_date.month
            except ValueError:
                # 如果格式错误，默认使用当前月份
                flash('月份格式错误，已查询当月数据。', 'warning')
                today = datetime.now()
                year = today.year
                month = today.month
        else:
            today = datetime.now()
            year = today.year
            month = today.month
            selected_month_str = today.strftime('%Y-%m') # 用于在模板中回显

        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            
            # 根据角色获取不同的统计数据
            if session.get('role') == 'admin':
                # 管理员查看所有用户的统计数据 (按选择的月份)
                cursor.execute("""
                    SELECT 
                        COUNT(DISTINCT DATE(check_in_time)) as total_days,
                        COUNT(DISTINCT CASE 
                            WHEN TIME(check_in_time) <= '09:00:00' 
                            THEN DATE(check_in_time) 
                        END) as normal_days,
                        COUNT(DISTINCT CASE 
                            WHEN TIME(check_in_time) > '09:00:00' 
                            THEN DATE(check_in_time)
                        END) as late_days,
                        COUNT(DISTINCT CASE 
                            WHEN check_out_time IS NULL OR TIME(check_out_time) < '18:00:00' 
                            THEN DATE(check_in_time)
                        END) as early_leave_days
                    FROM attendance
                    WHERE MONTH(check_in_time) = %s
                    AND YEAR(check_in_time) = %s
                """, (month, year))
                
                stats = cursor.fetchone()
                
                # 获取所有用户的情绪统计 (按选择的月份)
                cursor.execute("""
                    SELECT emotion, COUNT(*) as count
                    FROM attendance
                    WHERE MONTH(check_in_time) = %s
                    AND YEAR(check_in_time) = %s
                    AND emotion IS NOT NULL
                    GROUP BY emotion
                """, (month, year))
                
                emotion_stats = {row['emotion']: row['count'] for row in cursor.fetchall()}
                
                # 获取每日考勤人数统计 (按选择的月份)
                cursor.execute("""
                    SELECT 
                        DATE(check_in_time) as date,
                        COUNT(DISTINCT user_id) as count
                    FROM attendance
                    WHERE MONTH(check_in_time) = %s
                    AND YEAR(check_in_time) = %s
                    GROUP BY DATE(check_in_time)
                    ORDER BY date
                """, (month, year))
                
                attendance_trend = cursor.fetchall()
                
                # 获取用户出勤率排名 (按选择的月份)
                cursor.execute("""
                    SELECT 
                        u.name,
                        COUNT(DISTINCT DATE(a.check_in_time)) as attendance_days,
                        COUNT(DISTINCT CASE 
                            WHEN TIME(a.check_in_time) <= '09:00:00' 
                            THEN DATE(a.check_in_time)
                        END) as normal_days,
                        COUNT(DISTINCT CASE 
                            WHEN TIME(a.check_in_time) > '09:00:00' 
                            THEN DATE(a.check_in_time)
                        END) as late_days,
                        COUNT(DISTINCT CASE 
                            WHEN a.check_out_time IS NULL OR TIME(a.check_out_time) < '18:00:00' 
                            THEN DATE(a.check_in_time)
                        END) as early_leave_days
                    FROM users u
                    LEFT JOIN attendance a ON u.id = a.user_id
                    AND MONTH(a.check_in_time) = %s
                    AND YEAR(a.check_in_time) = %s
                    WHERE u.role = 'user'
                    GROUP BY u.id, u.name
                    ORDER BY attendance_days DESC, normal_days DESC
                """, (month, year))
                user_rankings = cursor.fetchall()
                
            else:
                # 普通用户只能查看自己的统计数据 (按选择的月份)
                user_id = session.get('user_id')
                cursor.execute("""
                    SELECT 
                        COUNT(DISTINCT DATE(check_in_time)) as total_days,
                        COUNT(DISTINCT CASE 
                            WHEN TIME(check_in_time) <= '09:00:00' 
                            THEN DATE(check_in_time)
                        END) as normal_days,
                        COUNT(DISTINCT CASE 
                            WHEN TIME(check_in_time) > '09:00:00' 
                            THEN DATE(check_in_time)
                        END) as late_days,
                        COUNT(DISTINCT CASE 
                            WHEN check_out_time IS NULL OR TIME(check_out_time) < '18:00:00' 
                            THEN DATE(check_in_time)
                        END) as early_leave_days
                    FROM attendance
                    WHERE user_id = %s
                    AND MONTH(check_in_time) = %s
                    AND YEAR(check_in_time) = %s
                """, (user_id, month, year))
                
                stats = cursor.fetchone()
                
                # 获取个人情绪统计 (按选择的月份)
                cursor.execute("""
                    SELECT emotion, COUNT(*) as count
                    FROM attendance
                    WHERE user_id = %s
                    AND MONTH(check_in_time) = %s
                    AND YEAR(check_in_time) = %s
                    AND emotion IS NOT NULL
                    GROUP BY emotion
                """, (user_id, month, year))
                
                emotion_stats = {row['emotion']: row['count'] for row in cursor.fetchall()}
                
                # 获取个人每日考勤数据 (按选择的月份)
                cursor.execute("""
                    SELECT DATE(check_in_time) as date,
                           COUNT(*) as count
                    FROM attendance
                    WHERE user_id = %s
                    AND MONTH(check_in_time) = %s
                    AND YEAR(check_in_time) = %s
                    GROUP BY DATE(check_in_time)
                    ORDER BY date
                """, (user_id, month, year))
                
                attendance_trend = cursor.fetchall()
                user_rankings = None
            
            # 处理日期和考勤数据
            dates = [row['date'].strftime('%Y-%m-%d') for row in attendance_trend]
            attendance_data = [row['count'] for row in attendance_trend]
            
            # 获取当前用户信息
            cursor.execute("SELECT name FROM users WHERE id = %s", (session.get('user_id'),))
            user = cursor.fetchone()
            
            # 确保关闭游标
            cursor.close()
            
            return render_template(
                'attendance_statistics.html',
                total_days=stats['total_days'] or 0,
                late_days=stats['late_days'] or 0,
                early_leave_days=stats['early_leave_days'] or 0,
                normal_days=stats['normal_days'] or 0,
                emotion_stats=emotion_stats,
                dates=dates,
                attendance_data=attendance_data,
                current_user=user,
                is_admin=session.get('role') == 'admin',
                user_rankings=user_rankings,
                selected_month=selected_month_str # 传递选择的月份用于回显
            )
            
    except Exception as e:
        logger.error(f"获取考勤统计失败: {str(e)}")
        flash('获取考勤统计失败', 'danger')
        # 确保重定向到正确的仪表盘
        redirect_url = url_for('admin.dashboard') if session.get('role') == 'admin' else url_for('user_dashboard')
        return redirect(redirect_url)

@attendance_bp.route('/get_attendance', methods=['GET'])
@login_required
def get_attendance():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)  # 使用字典游标
        
        # 管理员可以查看所有记录，普通用户只能查看自己的
        if session.get('role') == 'admin':
            cursor.execute("""
                SELECT a.name, DATE_FORMAT(a.check_in_time, '%%H:%%i:%%s') as time
                FROM attendance a
                WHERE DATE(a.check_in_time) = CURDATE()
                ORDER BY a.check_in_time DESC
            """)
        else:
            user_name = session.get('name')
            cursor.execute("""
                SELECT a.name, DATE_FORMAT(a.check_in_time, '%%H:%%i:%%s') as time
                FROM attendance a
                WHERE a.name = %s AND DATE(a.check_in_time) = CURDATE()
                ORDER BY a.check_in_time DESC
            """, (user_name,))
            
        records = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return jsonify({
            "status": "success",
            "records": records
        })
        
    except Exception as e:
        logger.error(f"获取考勤记录失败: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "获取考勤记录失败",
            "records": []
        })

@attendance_bp.route('/batch_check_in', methods=['POST'])
@login_required
def batch_check_in():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': '请先登录'})
    
    try:
        # 获取上传的图片
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': '未接收到图片'})
        
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        # 将图片转换为numpy数组
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 使用dlib进行人脸检测
        detector = dlib.get_frontal_face_detector()
        faces = detector(frame, 1)
        
        if not faces:
            return jsonify({'status': 'error', 'message': '未检测到人脸'})
            
        # 获取人脸特征提取器
        shape_predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
        face_rec_model = dlib.face_recognition_model_v1('data/data_dlib/dlib_face_recognition_resnet_model_v1.dat')
        
        # 获取所有已注册用户的人脸特征
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT u.id, u.name, k.face_encoding 
            FROM known_faces k
            JOIN users u ON k.user_id = u.id
            WHERE k.face_encoding IS NOT NULL
        """)
        registered_users = cursor.fetchall()
        
        if not registered_users:
            cursor.close()
            conn.close()
            return jsonify({'status': 'error', 'message': '没有注册用户'})
            
        # 获取当前日期
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # 记录识别结果
        success_list = []
        already_checked_list = []
        failed_list = []
        
        # 对每个检测到的人脸进行识别
        for face in faces:
            try:
                # 获取人脸特征
                shape = shape_predictor(frame, face)
                face_descriptor = face_rec_model.compute_face_descriptor(frame, shape)
                face_encoding = np.array(face_descriptor)
                
                # 初始化最佳匹配
                best_match = None
                best_distance = float('inf')
                best_user_id = None
                
                # 与所有注册用户比对
                for user_id, name, encoding_blob in registered_users:
                    known_encoding = np.frombuffer(encoding_blob, dtype=np.float64)
                    # 计算欧氏距离
                    distance = np.linalg.norm(face_encoding - known_encoding)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match = name
                        best_user_id = user_id
                
                # 如果找到匹配的人脸（使用阈值0.4）
                if best_distance <= 0.4:
                    # 检查今天是否已经打卡
                    cursor.execute("""
                        SELECT COUNT(*) FROM attendance 
                        WHERE user_id = %s AND DATE(check_in_time) = %s
                    """, (best_user_id, current_date))
                    
                    if cursor.fetchone()[0] > 0:
                        already_checked_list.append(best_match)
                    else:
                        # 添加考勤记录
                        cursor.execute("""
                            INSERT INTO attendance (user_id, name, check_in_time)
                            VALUES (%s, %s, NOW())
                        """, (best_user_id, best_match))
                        success_list.append(best_match)
                else:
                    failed_list.append("未知人员")
                    
            except Exception as e:
                logger.error(f"处理人脸时出错: {str(e)}")
                failed_list.append("处理失败")
                continue
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # 根据识别结果返回不同的消息
        if len(success_list) == 0 and len(already_checked_list) > 0:
            status = 'warning'
            message = '所有检测到的人员今日均已打卡'
        elif len(success_list) > 0:
            status = 'success'
            message = '打卡处理完成'
        else:
            status = 'error'
            message = '未能成功识别任何人员'
        
        return jsonify({
            'status': status,
            'message': message,
            'details': {
                'success': success_list,
                'already_checked': already_checked_list,
                'failed': failed_list
            }
        })
        
    except Exception as e:
        logger.error(f"批量打卡处理错误: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'处理失败: {str(e)}'
        })

@attendance_bp.route('/record', methods=['POST'])
@login_required
def record_attendance():
    try:
        with get_db_connection() as conn:
            with conn.cursor(dictionary=True) as cursor:
                user_id = request.form.get('user_id')
                if not user_id:
                    user_id = session['user_id']
                
                now = datetime.now()
                
                # 检查今日是否已打卡
                cursor.execute('''
                    SELECT COUNT(*) as count 
                    FROM attendance 
                    WHERE user_id = %s AND DATE(check_in_time) = %s
                ''', (user_id, now.date()))
                
                result = cursor.fetchone()
                if result['count'] > 0:
                    return jsonify({
                        'status': 'error',
                        'message': '今日已完成签到'
                    })
                
                # 记录考勤
                cursor.execute('''
                    INSERT INTO attendance (user_id, name, check_in_time) 
                    VALUES (%s, %s, %s)
                ''', (user_id, session.get('name'), now))
                conn.commit()
                
                return jsonify({
                    'status': 'success',
                    'message': '签到成功',
                    'time': now.strftime('%H:%M:%S')
                })
                
    except Exception as e:
        logger.error(f"记录考勤时发生错误: {e}")
        return jsonify({
            'status': 'error',
            'message': '签到失败，请稍后重试'
        }), 500 

@attendance_bp.route('/records')
@login_required
def records():
    """考勤记录页面"""
    try:
        if not db.connect():
            flash('数据库连接失败', 'danger')
            return render_template('attendance/records.html')
            
        # 获取考勤记录
        records = attendance.get_attendance_records(session['user_id'])
        
        # 获取考勤统计
        stats = attendance.get_statistics(session['user_id'])
        
        return render_template('attendance/records.html',
                             records=records,
                             stats=stats)
                             
    except Exception as e:
        logger.error(f"获取考勤记录失败: {str(e)}")
        flash('获取考勤记录失败，请稍后重试', 'danger')
        return render_template('attendance/records.html') 
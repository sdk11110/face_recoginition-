from flask import render_template, Response, session, redirect, url_for
import cv2
import dlib
import time
from models.face_recognition import face_recognizer
from models.attendance import attendance
from datetime import datetime

def video_feed():
    """生成视频流"""
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        return "无法打开摄像头"
        
    detector = dlib.get_frontal_face_detector()
    
    def generate():
        while True:
            success, frame = camera.read()
            if not success:
                break
                
            # 水平翻转
            frame = cv2.flip(frame, 1)
            
            # 检测人脸
            faces = detector(frame, 0)
            
            # 在检测到的人脸周围画框
            for face in faces:
                left = face.left()
                top = face.top()
                right = face.right()
                bottom = face.bottom()
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
            # 编码为JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.1)
            
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def check_in_page():
    """签到页面"""
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    return render_template('face_check_in.html', mode='check_in')

def check_out_page():
    """签退页面"""
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    return render_template('face_check_in.html', mode='check_out')

def attendance_query_page():
    """考勤查询页面"""
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    return render_template('attendance_query.html')

def attendance_statistics_page():
    """考勤统计页面"""
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    # 初始化所需的变量
    total_days = 0
    normal_days = 0
    late_days = 0
    early_leave_days = 0
    dates = []
    attendance_data = []
    emotion_stats = {}
    selected_month = datetime.now().strftime('%Y-%m')
    is_admin = session.get('role') == 'admin'
    user_rankings = []
    
    return render_template('attendance_statistics.html',
                          total_days=total_days,
                          normal_days=normal_days,
                          late_days=late_days,
                          early_leave_days=early_leave_days,
                          dates=dates,
                          attendance_data=attendance_data,
                          emotion_stats=emotion_stats,
                          selected_month=selected_month,
                          is_admin=is_admin,
                          user_rankings=user_rankings) 
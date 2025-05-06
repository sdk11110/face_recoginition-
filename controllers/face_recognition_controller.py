from flask import jsonify, request
from models.face_recognition import FaceRecognizer
from models.attendance import Attendance
from utils.face_utils import process_face_image, save_face_image
from utils.db_utils import get_user_by_id
import logging

def check_in():
    """处理打卡请求"""
    try:
        # 获取图像数据
        image_data = request.files['image'].read()
        
        # 处理人脸图像
        face_encoding, face_image = process_face_image(image_data)
        if not face_encoding:
            return jsonify({'success': False, 'message': '未检测到人脸'})
            
        # 获取人脸识别器实例
        face_recognizer = FaceRecognizer()
        
        # 识别人脸
        user_id = face_recognizer.recognize_face(face_encoding)
        if not user_id:
            return jsonify({'success': False, 'message': '未识别到用户'})
            
        # 获取用户信息
        user = get_user_by_id(user_id)
        if not user:
            return jsonify({'success': False, 'message': '用户不存在'})
            
        # 记录考勤
        attendance = Attendance()
        success = attendance.record_check_in(user_id, user['name'])
        
        if success:
            # 保存人脸图像
            save_face_image(face_image, user_id)
            return jsonify({'success': True, 'message': '打卡成功'})
        else:
            return jsonify({'success': False, 'message': '打卡失败'})
            
    except Exception as e:
        logging.error(f"打卡处理失败: {str(e)}")
        return jsonify({'success': False, 'message': '系统错误'})

def check_out():
    """处理签退请求"""
    try:
        # 获取图像数据
        image_data = request.files['image'].read()
        
        # 处理人脸图像
        face_encoding, face_image = process_face_image(image_data)
        if not face_encoding:
            return jsonify({'success': False, 'message': '未检测到人脸'})
            
        # 获取人脸识别器实例
        face_recognizer = FaceRecognizer()
        
        # 识别人脸
        user_id = face_recognizer.recognize_face(face_encoding)
        if not user_id:
            return jsonify({'success': False, 'message': '未识别到用户'})
            
        # 获取用户信息
        user = get_user_by_id(user_id)
        if not user:
            return jsonify({'success': False, 'message': '用户不存在'})
            
        # 记录签退
        attendance = Attendance()
        success = attendance.record_check_out(user_id)
        
        if success:
            return jsonify({'success': True, 'message': '签退成功'})
        else:
            return jsonify({'success': False, 'message': '签退失败'})
            
    except Exception as e:
        logging.error(f"签退处理失败: {str(e)}")
        return jsonify({'success': False, 'message': '系统错误'})

def get_attendance_records():
    """获取考勤记录"""
    try:
        user_id = request.args.get('user_id')
        attendance = Attendance()
        records = attendance.get_attendance_records(user_id)
        return jsonify({'success': True, 'records': records})
    except Exception as e:
        logging.error(f"获取考勤记录失败: {str(e)}")
        return jsonify({'success': False, 'message': '系统错误'})

def get_attendance_statistics():
    """获取考勤统计"""
    try:
        user_id = request.args.get('user_id')
        attendance = Attendance()
        statistics = attendance.get_statistics(user_id)
        return jsonify({'success': True, 'statistics': statistics})
    except Exception as e:
        logging.error(f"获取考勤统计失败: {str(e)}")
        return jsonify({'success': False, 'message': '系统错误'}) 
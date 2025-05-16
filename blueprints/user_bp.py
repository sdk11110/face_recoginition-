from flask import Blueprint, render_template, request, session, redirect, url_for, flash, jsonify
from datetime import datetime, timedelta
from utils.db_helper import get_db_connection
from utils.decorators import login_required
import logging

logger = logging.getLogger(__name__)

user_bp = Blueprint('user', __name__, url_prefix='/user')

@user_bp.route('/dashboard')
@login_required
def dashboard():
    """用户仪表盘"""
    # 这里可以添加获取用户特定数据的逻辑
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('auth.login'))
        
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # 修复：确保参数作为元组传递
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    
    # 获取最近打卡记录
    sql_query_recent_records = '''
        SELECT DATE(check_in_time) as check_date, TIME(check_in_time) as check_time
        FROM attendance
        WHERE user_id = %s
        ORDER BY check_in_time DESC
        LIMIT 5
    '''
    cursor.execute(sql_query_recent_records, (user_id,))
    recent_records = cursor.fetchall()
    
    # 获取本月打卡天数
    today = datetime.now()
    start_of_month = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    # 修复：确保参数作为元组传递
    cursor.execute("""
        SELECT COUNT(DISTINCT DATE(check_in_time))
        FROM attendance
        WHERE user_id = %s AND check_in_time >= %s
    """, (user_id, start_of_month))
    monthly_days = cursor.fetchone()['COUNT(DISTINCT DATE(check_in_time))']
    
    # 检查今天是否已打卡
    today_str = today.strftime('%Y-%m-%d')
    
    # 修复：确保参数作为元组传递
    cursor.execute("SELECT COUNT(*) FROM attendance WHERE user_id = %s AND DATE(check_in_time) = %s", 
                  (user_id, today_str))
    checked_today = cursor.fetchone()['COUNT(*)'] > 0
    
    cursor.close()
    conn.close()
    
    # 模拟已工作天数，实际应从数据库获取
    user['days_worked'] = 100 # 示例数据
    
    # 修复：避免在 strftime 中使用中文字符
    year = today.strftime('%Y')
    month = today.strftime('%m')
    day = today.strftime('%d')
    now_str = f"{year}年{month}月{day}日"
    
    return render_template('user-dashboard.html', 
                           user=user,
                           recent_records=recent_records,
                           monthly_days=monthly_days,
                           checked_today=checked_today,
                           now_str=now_str # 添加当前日期字符串
                           )

@user_bp.route('/attendance_statistics')
@login_required
def attendance_statistics():
    """用户个人考勤统计"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return redirect(url_for('auth.login'))
        
        # 获取用户选择的月份，默认为当前月份
        selected_month_str = request.args.get('month')
        if selected_month_str:
            try:
                selected_date = datetime.strptime(selected_month_str, '%Y-%m')
                year = selected_date.year
                month = selected_date.month
                selected_month_value = selected_month_str
            except ValueError:
                # 如果格式错误，默认使用当前月份
                flash('月份格式错误，已查询当月数据。', 'warning')
                today = datetime.now()
                year = today.year
                month = today.month
                selected_month_value = today.strftime('%Y-%m')
        else:
            today = datetime.now()
            year = today.year
            month = today.month
            selected_month_value = today.strftime('%Y-%m')
            
        selected_month = f"{year}年{month}月"
        
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            
            # 获取用户在选定月份的考勤统计
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
                    END) as early_days
                FROM attendance
                WHERE user_id = %s
                AND MONTH(check_in_time) = %s
                AND YEAR(check_in_time) = %s
            """, (user_id, month, year))
            
            stats = cursor.fetchone()
            
            # 获取用户在选定月份的情绪统计
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
            
            # 获取用户在选定月份的每日考勤状态
            # 首先获取月份的所有日期
            first_day = datetime(year, month, 1)
            if month == 12:
                last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
            else:
                last_day = datetime(year, month + 1, 1) - timedelta(days=1)
            
            dates = [(first_day + timedelta(days=i)).strftime('%Y-%m-%d') 
                    for i in range((last_day - first_day).days + 1)]
            
            # 获取用户在每一天的考勤记录
            cursor.execute("""
                SELECT 
                    DATE(check_in_time) as date,
                    CASE
                        WHEN TIME(check_in_time) <= '09:00:00' AND 
                             (check_out_time IS NOT NULL AND TIME(check_out_time) >= '18:00:00') 
                        THEN 1  -- 正常出勤 (准时上班和下班)
                        WHEN check_in_time IS NOT NULL THEN 0.5  -- 迟到或早退
                        ELSE 0  -- 缺勤
                    END as status
                FROM attendance
                WHERE user_id = %s
                AND MONTH(check_in_time) = %s
                AND YEAR(check_in_time) = %s
            """, (user_id, month, year))
            
            attendance_records = cursor.fetchall()
            
            # 构建日期到状态的映射
            attendance_map = {row['date'].strftime('%Y-%m-%d'): row['status'] for row in attendance_records}
            
            # 为每个日期分配状态值
            attendance_data = [attendance_map.get(date, 0) for date in dates]
            
        return render_template(
            'user_attendance_statistics.html',
            total_days=stats['total_days'] or 0,
            normal_days=stats['normal_days'] or 0,
            late_days=stats['late_days'] or 0,
            early_days=stats['early_days'] or 0,
            emotion_stats=emotion_stats,
            dates=dates,
            attendance_data=attendance_data,
            selected_month=selected_month,
            selected_month_value=selected_month_value
        )
        
    except Exception as e:
        logger.error(f"获取用户考勤统计失败: {str(e)}")
        flash('获取考勤统计失败', 'danger')
        return redirect(url_for('user.dashboard'))

@user_bp.route('/export_attendance')
@login_required
def export_attendance():
    """导出用户考勤记录"""
    try:
        user_id = session.get('user_id')
        selected_month = request.args.get('month', datetime.now().strftime('%Y-%m'))
        
        # 这里可以实现导出功能，例如生成CSV文件
        # 简单起见，这里只返回一个成功信息
        flash('考勤记录导出功能正在开发中', 'info')
        return redirect(url_for('user.attendance_statistics', month=selected_month))
        
    except Exception as e:
        logger.error(f"导出考勤记录失败: {str(e)}")
        flash('导出考勤记录失败', 'danger')
        return redirect(url_for('user.attendance_statistics')) 
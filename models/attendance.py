from datetime import datetime, time
import logging
from .database import db

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Attendance:
    def __init__(self):
        self.work_start_time = time(9, 0)  # 上班时间
        self.work_end_time = time(18, 0)   # 下班时间
        
    def record_check_in(self, user_id, name, emotion=None):
        """记录签到"""
        try:
            if not db.connect():
                return False
                
            # 检查今天是否已经签到
            today = datetime.now().date()
            db.cursor.execute("""
                SELECT id FROM attendance 
                WHERE user_id = %s AND DATE(check_in_time) = %s
            """, (user_id, today))
            
            if db.cursor.fetchone():
                logger.warning(f"用户 {name} 今天已经签到")
                return False
                
            # 记录签到
            check_in_time = datetime.now()
            db.cursor.execute("""
                INSERT INTO attendance (user_id, name, check_in_time, emotion)
                VALUES (%s, %s, %s, %s)
            """, (user_id, name, check_in_time, emotion))
            
            db.conn.commit()
            logger.info(f"用户 {name} 签到成功")
            return True
            
        except Exception as e:
            logger.error(f"签到记录失败: {str(e)}")
            if db.conn:
                db.conn.rollback()
            return False
        finally:
            db.close()
            
    def record_check_out(self, user_id, name):
        """记录签退"""
        try:
            if not db.connect():
                return False
                
            # 检查今天是否已经签到
            today = datetime.now().date()
            db.cursor.execute("""
                SELECT id, check_in_time FROM attendance 
                WHERE user_id = %s AND DATE(check_in_time) = %s
            """, (user_id, today))
            
            record = db.cursor.fetchone()
            if not record:
                logger.warning(f"用户 {name} 今天尚未签到")
                return False
                
            # 检查是否已经签退
            if record[1] is not None:
                logger.warning(f"用户 {name} 今天已经签退")
                return False
                
            # 记录签退
            check_out_time = datetime.now()
            db.cursor.execute("""
                UPDATE attendance 
                SET check_out_time = %s
                WHERE id = %s
            """, (check_out_time, record[0]))
            
            db.conn.commit()
            logger.info(f"用户 {name} 签退成功")
            return True
            
        except Exception as e:
            logger.error(f"签退记录失败: {str(e)}")
            if db.conn:
                db.conn.rollback()
            return False
        finally:
            db.close()
            
    def get_attendance_records(self, user_id=None, start_date=None, end_date=None):
        """获取考勤记录"""
        try:
            if not db.connect():
                return []
                
            query = """
                SELECT a.id, a.user_id, a.name, a.check_in_time, 
                       a.check_out_time, a.emotion, a.created_at
                FROM attendance a
            """
            params = []
            
            if user_id:
                query += " WHERE a.user_id = %s"
                params.append(user_id)
                
            if start_date:
                if not user_id:
                    query += " WHERE"
                else:
                    query += " AND"
                query += " DATE(a.check_in_time) >= %s"
                params.append(start_date)
                
            if end_date:
                if not user_id and not start_date:
                    query += " WHERE"
                else:
                    query += " AND"
                query += " DATE(a.check_in_time) <= %s"
                params.append(end_date)
                
            query += " ORDER BY a.check_in_time DESC"
            
            db.cursor.execute(query, params)
            records = []
            
            for row in db.cursor.fetchall():
                record = {
                    'id': row[0],
                    'user_id': row[1],
                    'name': row[2],
                    'check_in_time': row[3],
                    'check_out_time': row[4],
                    'emotion': row[5],
                    'created_at': row[6]
                }
                records.append(record)
                
            return records
            
        except Exception as e:
            logger.error(f"获取考勤记录失败: {str(e)}")
            return []
        finally:
            db.close()
            
    def get_statistics(self, user_id=None, start_date=None, end_date=None):
        """获取考勤统计"""
        try:
            records = self.get_attendance_records(user_id, start_date, end_date)
            
            total_days = len(records)
            normal_days = 0
            late_days = 0
            early_leave_days = 0
            
            for record in records:
                check_in_time = record['check_in_time']
                check_out_time = record['check_out_time']
                
                if check_in_time and check_in_time.time() > self.work_start_time:
                    late_days += 1
                elif check_out_time and check_out_time.time() < self.work_end_time:
                    early_leave_days += 1
                else:
                    normal_days += 1
                    
            return {
                'total_days': total_days,
                'normal_days': normal_days,
                'late_days': late_days,
                'early_leave_days': early_leave_days
            }
            
        except Exception as e:
            logger.error(f"获取考勤统计失败: {str(e)}")
            return {
                'total_days': 0,
                'normal_days': 0,
                'late_days': 0,
                'early_leave_days': 0
            }

# 创建考勤实例
attendance = Attendance() 
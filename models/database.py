import mysql.connector
import os
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MySQL配置
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',  # MySQL密码
    'database': 'attendance_db'
}

class Database:
    def __init__(self):
        self.conn = None
        self.cursor = None
        
    def connect(self):
        try:
            self.conn = mysql.connector.connect(**db_config)
            self.cursor = self.conn.cursor()
            return True
        except Exception as e:
            logger.error(f"数据库连接失败: {str(e)}")
            return False
            
    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            
    def init_db(self):
        try:
            if not self.connect():
                return False
                
            # 创建数据库
            self.cursor.execute("CREATE DATABASE IF NOT EXISTS attendance_db")
            self.cursor.execute("USE attendance_db")
            
            # 创建用户表
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    name VARCHAR(100) NOT NULL,
                    role ENUM('admin', 'user') NOT NULL DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            
            # 创建考勤表
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    name VARCHAR(100) NOT NULL,
                    check_in_time TIMESTAMP,
                    check_out_time TIMESTAMP,
                    emotion VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            # 创建已知人脸表
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS known_faces (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    face_encoding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            # 检查是否需要添加emotion字段
            self.cursor.execute("SHOW COLUMNS FROM attendance LIKE 'emotion'")
            if not self.cursor.fetchone():
                self.cursor.execute("ALTER TABLE attendance ADD COLUMN emotion VARCHAR(20)")
                logger.info("已添加emotion字段到attendance表")
            
            self.conn.commit()
            logger.info("数据库初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {str(e)}")
            if self.conn:
                self.conn.rollback()
            return False
        finally:
            self.close()
            
    def get_next_id(self, table_name):
        try:
            if not self.connect():
                return None
                
            self.cursor.execute(f"SELECT MAX(id) FROM {table_name}")
            result = self.cursor.fetchone()
            return (result[0] + 1) if result[0] else 1
            
        except Exception as e:
            logger.error(f"获取下一个ID失败: {str(e)}")
            return None
        finally:
            self.close()

# 创建数据库实例
db = Database() 
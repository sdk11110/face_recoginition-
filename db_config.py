import mysql.connector
from mysql.connector import pooling
import logging
import os
from contextlib import contextmanager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 从环境变量获取数据库配置
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', '123456'),
    'database': os.getenv('DB_NAME', 'attendance_db'),
    'auth_plugin': 'mysql_native_password',
    'pool_name': 'mypool',
    'pool_size': 5
}

# 创建连接池
try:
    connection_pool = mysql.connector.pooling.MySQLConnectionPool(**DB_CONFIG)
    logger.info("数据库连接池创建成功")
except Exception as e:
    logger.error(f"创建数据库连接池失败: {e}")
    raise

@contextmanager
def get_db_connection():
    """
    使用上下文管理器获取数据库连接
    使用方法:
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
    """
    conn = None
    try:
        conn = connection_pool.get_connection()
        yield conn
    except Exception as e:
        logger.error(f"数据库连接错误: {e}")
        if conn and conn.is_connected():
            conn.rollback()
        raise
    finally:
        if conn and conn.is_connected():
            conn.close()
            logger.debug("数据库连接已关闭")

def init_db():
    """
    初始化数据库表结构
    """
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            try:
                # 创建用户表
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) NOT NULL UNIQUE,
                    password VARCHAR(100) NOT NULL,
                    name VARCHAR(100) NOT NULL,
                    role ENUM('admin', 'user') NOT NULL DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
                ''')
                
                # 创建考勤记录表
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    name VARCHAR(100) NOT NULL,
                    check_in_time DATETIME NOT NULL,
                    check_out_time DATETIME DEFAULT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                ''')
                
                # 创建人脸编码表
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS known_faces (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    name VARCHAR(100) NOT NULL,
                    face_encoding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                ''')
                
                conn.commit()
                logger.info("数据库表初始化完成")
                
            except Exception as e:
                logger.error(f"初始化数据库失败: {e}")
                raise 
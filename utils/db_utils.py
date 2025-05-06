import mysql.connector
import logging
from models.database import Database

def get_db_connection():
    """获取数据库连接"""
    try:
        db = Database()
        return db.get_connection()
    except Exception as e:
        logging.error(f"获取数据库连接失败: {str(e)}")
        return None

def execute_query(query, params=None, fetch_one=False):
    """执行数据库查询"""
    conn = get_db_connection()
    if not conn:
        return None
        
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, params or ())
        
        if fetch_one:
            result = cursor.fetchone()
        else:
            result = cursor.fetchall()
            
        return result
    except Exception as e:
        logging.error(f"执行查询失败: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

def execute_update(query, params=None):
    """执行数据库更新操作"""
    conn = get_db_connection()
    if not conn:
        return False
        
    try:
        cursor = conn.cursor()
        cursor.execute(query, params or ())
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"执行更新失败: {str(e)}")
        conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def get_user_by_id(user_id):
    """根据用户ID获取用户信息"""
    query = "SELECT * FROM users WHERE id = %s"
    return execute_query(query, (user_id,), fetch_one=True)

def get_user_by_username(username):
    """根据用户名获取用户信息"""
    query = "SELECT * FROM users WHERE username = %s"
    return execute_query(query, (username,), fetch_one=True)

def insert_user(username, password, name, role='user'):
    """插入新用户"""
    query = """
    INSERT INTO users (username, password, name, role)
    VALUES (%s, %s, %s, %s)
    """
    return execute_update(query, (username, password, name, role))

def update_user_password(user_id, new_password):
    """更新用户密码"""
    query = "UPDATE users SET password = %s WHERE id = %s"
    return execute_update(query, (new_password, user_id))

def delete_user(user_id):
    """删除用户"""
    query = "DELETE FROM users WHERE id = %s"
    return execute_update(query, (user_id,)) 
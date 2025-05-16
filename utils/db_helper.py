import mysql.connector
import logging
from db_config import connection_pool

# 配置日志
logger = logging.getLogger(__name__)

def get_db_connection():
    """
    获取数据库连接
    返回一个可用的数据库连接对象
    """
    try:
        conn = connection_pool.get_connection()
        return conn
    except Exception as e:
        logger.error(f"获取数据库连接失败: {str(e)}")
        raise 
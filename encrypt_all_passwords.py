import mysql.connector
from werkzeug.security import generate_password_hash

# 数据库连接配置
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='123456',  # 修改为你的MySQL密码
    database='attendance_db'
)
cursor = conn.cursor()

# 查询所有用户的id和明文密码
cursor.execute("SELECT id, password FROM users")
users = cursor.fetchall()

for user_id, pwd in users:
    # 如果已经是哈希值则跳过（简单判断）
    if pwd.startswith('pbkdf2:sha256:'):
        continue
    hashed = generate_password_hash(pwd)
    cursor.execute("UPDATE users SET password=%s WHERE id=%s", (hashed, user_id))
    print(f"用户ID {user_id} 密码已加密")

conn.commit()
cursor.close()
conn.close()
print("所有密码已加密！") 
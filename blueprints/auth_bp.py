from flask import Blueprint, render_template, request, jsonify, redirect, url_for, session, flash
import logging
from models.database import db
from werkzeug.security import check_password_hash, generate_password_hash

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建蓝图
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

# 登录页面
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """登录页面"""
    if request.method == 'GET':
        # Redirect if already logged in
        if 'user_id' in session:
            # We need placeholders for admin/user dashboards until those blueprints are created
            # Using temporary direct URL strings might be safer for now if url_for fails
            if session.get('role') == 'admin':
                 # return redirect(url_for('admin_dashboard')) # Use main app route for now
                 return redirect('/admin_dashboard')
            else:
                 # return redirect(url_for('user_dashboard')) # Use main app route for now
                 return redirect('/user_dashboard')
        return render_template('login.html')

    if request.method == 'POST':
        try:
            username = request.form.get('username')
            password = request.form.get('password')
            
            if not username or not password:
                flash('请填写用户名和密码', 'danger')
                return render_template('login.html')
                
            if not db.connect():
                flash('数据库连接失败', 'danger')
                return render_template('login.html')
                
            # 查询用户
            db.cursor.execute("""
                SELECT id, username, password, name, role 
                FROM users 
                WHERE username = %s AND is_active = TRUE
            """, (username,))
            
            user = db.cursor.fetchone()
            db.close()
            
            if not user:
                flash('用户名或密码错误', 'danger')
                return render_template('login.html')
                
            # 验证密码
            if not check_password_hash(user[2], password):
                flash('用户名或密码错误', 'danger')
                return render_template('login.html')
                
            # 设置会话
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['name'] = user[3]
            session['role'] = user[4]
            
            flash('登录成功', 'success')
            # 根据角色跳转到不同的仪表板
            if session['role'] == 'admin':
                return redirect(url_for('admin.dashboard')) # 使用 admin 蓝图的 dashboard
            else:
                return redirect(url_for('user_dashboard')) # 使用 app 下的 user_dashboard
            
        except Exception as e:
            logger.error(f"登录失败: {str(e)}")
            flash('登录失败，请稍后重试', 'danger')
            return render_template('login.html')

# 登出
@auth_bp.route('/logout')
def logout():
    """退出登录"""
    session.clear()
    flash('已退出登录', 'success')
    return redirect(url_for('auth.login'))

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """注册页面"""
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            password = request.form.get('password')
            name = request.form.get('name')
            
            if not all([username, password, name]):
                flash('请填写所有必填字段', 'danger')
                return render_template('register.html')
                
            if not db.connect():
                flash('数据库连接失败', 'danger')
                return render_template('register.html')
                
            # 检查用户名是否已存在
            db.cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
            if db.cursor.fetchone():
                flash('用户名已存在', 'danger')
                db.close()
                return render_template('register.html')
                
            # 添加新用户
            hashed_password = generate_password_hash(password)
            db.cursor.execute("""
                INSERT INTO users (username, password, name, role)
                VALUES (%s, %s, %s, 'user')
            """, (username, hashed_password, name))
            
            db.conn.commit()
            db.close()
            
            flash('注册成功，请登录', 'success')
            return redirect(url_for('auth.login'))
            
        except Exception as e:
            logger.error(f"注册失败: {str(e)}")
            flash('注册失败，请稍后重试', 'danger')
            return render_template('register.html')
            
    return render_template('register.html') 
import functools
from flask import session, flash, redirect, url_for

# 装饰器：要求用户登录
def login_required(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            flash('请先登录以访问此页面', 'warning')
            # Point to the login route in the auth blueprint
            return redirect(url_for('auth.login'))
        return func(*args, **kwargs)
    return wrapper

# 装饰器：只允许管理员访问
def admin_required(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # First check if logged in
        if 'user_id' not in session:
            flash('请先登录以访问此页面', 'warning')
            return redirect(url_for('auth.login'))
        # Then check role
        if session.get('role') != 'admin':
            flash('需要管理员权限才能访问此页面', 'danger')
            # Redirect non-admins to their dashboard (assuming a user blueprint exists)
            # If user blueprint doesn't exist yet, redirecting to auth.login might be safer
            # Or maybe redirect to the main index which handles role-based redirection? Let's try user dashboard first.
            # This requires the user blueprint to exist.
            # return redirect(url_for('user.user_dashboard')) # Needs user_bp, comment out for now
            # Redirecting to index might be better until user_bp exists
            return redirect(url_for('index')) 
        return func(*args, **kwargs)
    return wrapper 
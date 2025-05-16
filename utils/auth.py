import functools
from flask import session, redirect, url_for, flash

def login_required(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            flash('请先登录以访问此页面', 'warning')
            return redirect(url_for('auth.login'))
        return func(*args, **kwargs)
    return wrapper

def admin_required(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            flash('请先登录以访问此页面', 'warning')
            return redirect(url_for('auth.login'))
        if session.get('role') != 'admin':
            flash('需要管理员权限才能访问此页面', 'danger')
            return redirect(url_for('user_dashboard'))
        return func(*args, **kwargs)
    return wrapper 
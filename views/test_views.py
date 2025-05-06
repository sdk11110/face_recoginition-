from flask import render_template

def test_page():
    """测试页面视图"""
    return render_template('test.html') 
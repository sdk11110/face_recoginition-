from flask import jsonify, request
import logging

def test_api():
    """测试API"""
    try:
        data = request.get_json()
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        logging.error(f"测试API失败: {str(e)}")
        return jsonify({'success': False, 'message': '系统错误'}) 
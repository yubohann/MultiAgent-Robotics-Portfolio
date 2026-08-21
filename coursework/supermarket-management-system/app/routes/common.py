from functools import wraps

from flask import flash, jsonify, redirect, request, session, url_for


def login_required(f):
    """登录验证装饰器"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('请先登录', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


def role_required(*roles):
    """角色验证装饰器。"""

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                if request.path.startswith('/api/'):
                    return jsonify({'success': False, 'message': '请先登录'}), 401
                flash('请先登录', 'error')
                return redirect(url_for('login'))

            user_role = session.get('role')
            if user_role not in roles:
                if request.path.startswith('/api/'):
                    return jsonify({'success': False, 'message': '无权限访问'}), 403
                flash('无权限访问', 'error')
                if user_role == 'cashier':
                    return redirect(url_for('cashier_page'))
                return redirect(url_for('index'))

            return f(*args, **kwargs)

        return decorated_function

    return decorator


def cashier_required(f):
    """仅收银员可访问。"""
    return role_required('cashier')(f)


def admin_required(f):
    """仅管理员可访问。"""
    return role_required('admin')(f)

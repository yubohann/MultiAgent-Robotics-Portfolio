from datetime import datetime

from werkzeug.security import generate_password_hash

from app import db
from app.models import AdminSignupRequest, User


def register_user(username, password, confirm_password, role='cashier'):
    """
    用户注册

    Args:
        username: 用户名
        password: 密码
        confirm_password: 确认密码
        role: 角色（cashier/admin）

    Returns:
        tuple: (success: bool, message: str)
    """
    username = (username or '').strip()
    role = (role or 'cashier').strip().lower()

    if not username or not password:
        return False, '用户名和密码不能为空'

    if password != confirm_password:
        return False, '两次密码不一致'

    if role not in ('cashier', 'admin'):
        return False, '请选择有效身份'

    if User.query.filter_by(username=username).first():
        return False, '用户名已存在'

    existing_request = AdminSignupRequest.query.filter_by(username=username).first()

    if role == 'cashier':
        if existing_request and existing_request.status == 'pending':
            return False, '该用户名已有管理员注册申请，请更换用户名'

        try:
            user = User(username=username, role='cashier', is_active=1)
            user.set_password(password)

            db.session.add(user)
            db.session.commit()

            return True, '收银员注册成功'
        except Exception as e:
            db.session.rollback()
            return False, f'注册失败：{str(e)}'

    try:
        hashed_password = generate_password_hash(password)

        if existing_request:
            if existing_request.status == 'pending':
                return False, '管理员申请已提交，请等待审核'
            if existing_request.status == 'approved':
                return False, '该管理员账号已审核通过，请直接登录'

            existing_request.password_hash = hashed_password
            existing_request.status = 'pending'
            existing_request.reviewed_by = None
            existing_request.reviewed_at = None
            existing_request.reject_reason = None
            existing_request.created_at = datetime.now()
        else:
            admin_request = AdminSignupRequest(
                username=username,
                password_hash=hashed_password,
                status='pending',
            )
            db.session.add(admin_request)

        db.session.commit()
        return True, '管理员注册申请已提交，待管理员审核后可登录'
    except Exception as e:
        db.session.rollback()
        return False, f'注册失败：{str(e)}'


def login_user(username, password):
    """
    用户登录

    Args:
        username: 用户名
        password: 密码

    Returns:
        tuple: (success: bool, message: str, user: User or None)
    """
    if not username or not password:
        return False, '用户名和密码不能为空', None

    username = (username or '').strip()
    user = User.query.filter_by(username=username).first()

    if not user:
        pending_request = AdminSignupRequest.query.filter_by(username=username, status='pending').first()
        if pending_request:
            return False, '管理员账号审核中，请联系管理员审批', None
        return False, '用户名或密码错误', None

    if not user.is_active:
        return False, '账号已被禁用', None

    if not user.check_password(password):
        return False, '用户名或密码错误', None

    return True, '登录成功', user


def get_admin_signup_requests(status='pending'):
    """获取管理员注册申请列表。"""
    query = AdminSignupRequest.query
    if status in ('pending', 'approved', 'rejected'):
        query = query.filter_by(status=status)
    return query.order_by(AdminSignupRequest.created_at.desc()).all()


def review_admin_signup_request(request_id, reviewer_id, decision, reject_reason=None):
    """管理员审核注册申请。"""
    if decision not in ('approve', 'reject'):
        return False, '无效的审核操作'

    reviewer = User.query.filter_by(user_id=reviewer_id, role='admin', is_active=1).first()
    if not reviewer:
        return False, '审核人无权限或已被禁用'

    signup_request = db.session.get(AdminSignupRequest, request_id)
    if not signup_request:
        return False, '申请记录不存在'

    if signup_request.status != 'pending':
        return False, '该申请已处理，请刷新页面后重试'

    try:
        if decision == 'approve':
            if User.query.filter_by(username=signup_request.username).first():
                return False, '用户名已存在，无法通过审核'

            new_admin = User(
                username=signup_request.username,
                role='admin',
                is_active=1,
            )
            new_admin.password_hash = signup_request.password_hash
            db.session.add(new_admin)

            signup_request.status = 'approved'
            signup_request.reject_reason = None
            success_message = '审核通过，管理员账号已创建'
        else:
            cleaned_reason = (reject_reason or '').strip()
            signup_request.status = 'rejected'
            signup_request.reject_reason = cleaned_reason[:200] if cleaned_reason else '未通过审核'
            success_message = '已拒绝管理员注册申请'

        signup_request.reviewed_by = reviewer.user_id
        signup_request.reviewed_at = datetime.now()

        db.session.commit()
        return True, success_message
    except Exception as e:
        db.session.rollback()
        return False, f'审核失败：{str(e)}'

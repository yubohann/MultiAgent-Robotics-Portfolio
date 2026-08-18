from app.models import AdminSignupRequest, User
from app.services.auth import login_user, register_user, review_admin_signup_request
from tests.conftest import login_as


def test_login_and_admin_signup_review(app, users):
    with app.app_context():
        success, message, user = login_user('admin', 'admin123')
        assert success is True
        assert message == '登录成功'
        assert user.role == 'admin'

        success, message, user = login_user('admin', 'wrong')
        assert success is False
        assert user is None

        success, message = register_user('newadmin', 'pw123456', 'pw123456', 'admin')
        assert success is True
        assert '待管理员审核' in message
        request = AdminSignupRequest.query.filter_by(username='newadmin').first()
        assert request is not None

        success, message = review_admin_signup_request(request.request_id, users['admin_id'], 'approve')
        assert success is True
        assert '管理员账号已创建' in message
        assert User.query.filter_by(username='newadmin', role='admin').first() is not None


def test_admin_api_requires_login_and_role(client, users):
    response = client.get('/api/members')
    assert response.status_code == 401
    assert response.get_json()['message'] == '请先登录'

    login_as(client, users['cashier_id'], username='cashier01', role='cashier')
    response = client.get('/api/members')
    assert response.status_code == 403
    assert response.get_json()['message'] == '无权限访问'

    login_as(client, users['admin_id'])
    response = client.get('/api/members')
    assert response.status_code == 200
    assert response.get_json()['items'] == []


def test_admin_pages_render_for_second_phase_modules(client, users):
    login_as(client, users['admin_id'])
    for path, title in [
        ('/members', '会员管理'),
        ('/employees', '员工管理'),
        ('/suppliers', '供应商管理'),
        ('/system-settings', '系统管理'),
    ]:
        response = client.get(path)
        assert response.status_code == 200
        assert title.encode('utf-8') in response.data


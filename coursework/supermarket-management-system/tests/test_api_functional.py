from tests.conftest import login_as


def test_second_phase_api_crud_flow(client, users):
    login_as(client, users['admin_id'])

    response = client.post('/api/members', json={
        'member_no': 'M8001',
        'member_name': '接口会员',
        'phone': '13880010001',
        'level': 'normal',
        'points': 10,
    })
    assert response.status_code == 200
    assert response.get_json()['success'] is True

    response = client.get('/api/members?search=M8001')
    member = response.get_json()['items'][0]
    assert member['member_name'] == '接口会员'

    response = client.put(f"/api/members/{member['member_id']}", json={'level': 'vip'})
    assert response.get_json()['success'] is True

    response = client.post(f"/api/members/{member['member_id']}/points", json={
        'points_delta': 5,
        'reason': '接口测试',
    })
    assert response.get_json()['success'] is True

    response = client.post(f"/api/members/{member['member_id']}/status", json={'status': 'inactive'})
    assert response.get_json()['success'] is True

    response = client.post('/api/employees', json={
        'employee_no': 'E8001',
        'employee_name': '接口员工',
        'position': '收银员',
    })
    assert response.get_json()['success'] is True
    employee = client.get('/api/employees?search=E8001').get_json()['items'][0]
    assert employee['position'] == '收银员'

    response = client.post('/api/suppliers', json={
        'supplier_code': 'S8001',
        'supplier_name': '接口供应商',
        'settlement_cycle': 'monthly',
    })
    assert response.get_json()['success'] is True
    supplier = client.get('/api/suppliers?search=S8001').get_json()['items'][0]
    assert supplier['supplier_name'] == '接口供应商'

    response = client.post('/api/system-settings', json={
        'setting_key': 'test.key',
        'setting_value': 'enabled',
        'description': '接口测试参数',
    })
    assert response.get_json()['success'] is True
    setting = client.get('/api/system-settings?search=test.key').get_json()['items'][0]
    assert setting['setting_value'] == 'enabled'


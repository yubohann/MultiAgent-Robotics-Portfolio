from app.services.second_phase import (
    adjust_member_points,
    create_employee,
    create_member,
    create_supplier,
    get_employees,
    get_members,
    get_suppliers,
    get_system_settings,
    set_employee_status,
    set_member_status,
    set_supplier_status,
    update_employee,
    update_member,
    update_supplier,
    upsert_system_setting,
)


def assert_failed(result):
    success, message = result
    assert success is False
    assert isinstance(message, str)


def test_member_management_flow(app):
    with app.app_context():
        success, message = create_member({
            'member_no': 'M9001',
            'member_name': '测试会员',
            'phone': '13890010001',
            'level': 'silver',
            'points': 100,
        })
        assert success is True
        assert message == '会员创建成功'
        member = get_members(search='测试会员')['items'][0]
        assert member['points'] == 100

        success, message = update_member(member['member_id'], {'level': 'gold', 'phone': '13890010002'})
        assert success is True
        assert get_members(search='M9001')['items'][0]['level'] == 'gold'

        success, message = adjust_member_points(member['member_id'], 50, '消费赠送')
        assert success is True
        assert get_members(search='M9001')['items'][0]['points'] == 150

        success, message = set_member_status(member['member_id'], 'inactive')
        assert success is True
        assert get_members(status='inactive')['items'][0]['status'] == 'inactive'


def test_employee_supplier_and_system_setting_flow(app):
    with app.app_context():
        success, message = create_employee({
            'employee_no': 'E9001',
            'employee_name': '测试员工',
            'position': '库管员',
            'phone': '13990010001',
            'work_schedule': '早班',
        })
        assert success is True
        employee = get_employees(search='测试员工')['items'][0]
        assert employee['position'] == '库管员'

        success, message = update_employee(employee['employee_id'], {'position': '收银员'})
        assert success is True
        assert get_employees(search='E9001')['items'][0]['position'] == '收银员'

        success, message = set_employee_status(employee['employee_id'], 'inactive')
        assert success is True
        assert get_employees(status='inactive')['items'][0]['employee_no'] == 'E9001'

        success, message = create_supplier({
            'supplier_code': 'S9001',
            'supplier_name': '测试供应商',
            'contact_person': '王经理',
            'phone': '027-90010001',
            'settlement_cycle': 'monthly',
        })
        assert success is True
        supplier = get_suppliers(search='测试供应商')['items'][0]
        assert supplier['settlement_cycle'] == 'monthly'

        success, message = update_supplier(supplier['supplier_id'], {'settlement_cycle': 'weekly'})
        assert success is True
        assert get_suppliers(search='S9001')['items'][0]['settlement_cycle'] == 'weekly'

        success, message = set_supplier_status(supplier['supplier_id'], 'inactive')
        assert success is True
        assert get_suppliers(status='inactive')['items'][0]['supplier_code'] == 'S9001'

        success, message = upsert_system_setting({
            'setting_key': 'receipt.footer',
            'setting_value': '欢迎下次光临',
            'description': '小票页脚',
        })
        assert success is True
        assert get_system_settings(search='receipt.footer')['items'][0]['setting_value'] == '欢迎下次光临'

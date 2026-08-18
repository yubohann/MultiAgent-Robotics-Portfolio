from app.services.second_phase import (
    adjust_member_points,
    create_employee,
    create_member,
    create_supplier,
    get_employees,
    get_members,
    get_suppliers,
    set_employee_status,
    update_employee,
    update_member,
    update_supplier,
    upsert_system_setting,
)


def assert_failed(result):
    success, message = result
    assert success is False
    assert isinstance(message, str)


def test_second_phase_member_error_paths(app):
    with app.app_context():
        assert_failed(create_member({'member_no': '', 'member_name': ''}))
        assert_failed(create_member({'member_no': 'M9101', 'member_name': 'A', 'level': 'diamond'}))
        assert_failed(create_member({'member_no': 'M9102', 'member_name': 'A', 'status': 'paused'}))

        assert create_member({
            'member_no': 'M9103',
            'member_name': 'Alice',
            'phone': '13891030001',
            'level': 'normal',
            'points': 20,
        })[0] is True
        assert_failed(create_member({'member_no': 'M9103', 'member_name': 'Duplicate'}))
        assert_failed(create_member({'member_no': 'M9104', 'member_name': 'PhoneDup', 'phone': '13891030001'}))

        assert create_member({
            'member_no': 'M9105',
            'member_name': 'Bob',
            'phone': '13891050001',
        })[0] is True
        first = get_members(search='M9103')['items'][0]
        second = get_members(search='M9105')['items'][0]

        assert_failed(update_member(9999, {'member_name': 'Missing'}))
        assert_failed(update_member(first['member_id'], {'member_no': ''}))
        assert_failed(update_member(first['member_id'], {'member_no': 'M9105'}))
        assert_failed(update_member(first['member_id'], {'phone': '13891050001'}))
        assert_failed(update_member(first['member_id'], {'level': 'diamond'}))
        assert_failed(update_member(first['member_id'], {'status': 'paused'}))

        assert update_member(first['member_id'], {
            'member_no': 'M9106',
            'member_name': 'Alice Updated',
            'phone': '',
            'points': -5,
        })[0] is True
        updated = get_members(search='M9106')['items'][0]
        assert updated['member_name'] == 'Alice Updated'
        assert updated['phone'] == ''
        assert updated['points'] == 0

        assert_failed(adjust_member_points(9999, 1))
        assert_failed(adjust_member_points(second['member_id'], -1_000))
        assert_failed(adjust_member_points(second['member_id'], 'bad-number'))


def test_second_phase_employee_error_paths(app):
    with app.app_context():
        assert_failed(create_employee({'employee_no': '', 'employee_name': '', 'position': ''}))
        assert_failed(create_employee({
            'employee_no': 'E9101',
            'employee_name': 'Alice',
            'position': 'Cashier',
            'status': 'paused',
        }))

        assert create_employee({
            'employee_no': 'E9102',
            'employee_name': 'Alice',
            'position': 'Cashier',
            'phone': '13991020001',
            'work_schedule': 'Morning',
        })[0] is True
        assert_failed(create_employee({'employee_no': 'E9102', 'employee_name': 'Duplicate', 'position': 'Cashier'}))
        assert_failed(create_employee({
            'employee_no': 'E9103',
            'employee_name': 'PhoneDup',
            'position': 'Cashier',
            'phone': '13991020001',
        }))

        assert create_employee({
            'employee_no': 'E9104',
            'employee_name': 'Bob',
            'position': 'Keeper',
            'phone': '13991040001',
        })[0] is True
        first = get_employees(search='E9102')['items'][0]
        second = get_employees(search='E9104')['items'][0]

        assert_failed(update_employee(9999, {'employee_name': 'Missing'}))
        assert_failed(update_employee(first['employee_id'], {'employee_no': ''}))
        assert_failed(update_employee(first['employee_id'], {'employee_no': 'E9104'}))
        assert_failed(update_employee(first['employee_id'], {'phone': '13991040001'}))
        assert_failed(update_employee(first['employee_id'], {'status': 'paused'}))

        assert update_employee(first['employee_id'], {
            'employee_no': 'E9105',
            'employee_name': 'Alice Updated',
            'position': 'Manager',
            'phone': '',
            'work_schedule': 'Night',
        })[0] is True
        updated = get_employees(search='E9105')['items'][0]
        assert updated['employee_name'] == 'Alice Updated'
        assert updated['position'] == 'Manager'
        assert updated['work_schedule'] == 'Night'
        assert set_employee_status(second['employee_id'], 'inactive')[0] is True


def test_second_phase_supplier_and_setting_error_paths(app):
    with app.app_context():
        assert_failed(create_supplier({'supplier_code': '', 'supplier_name': ''}))
        assert_failed(create_supplier({'supplier_code': 'S9101', 'supplier_name': 'A', 'settlement_cycle': 'yearly'}))
        assert_failed(create_supplier({'supplier_code': 'S9102', 'supplier_name': 'A', 'status': 'paused'}))

        assert create_supplier({
            'supplier_code': 'S9103',
            'supplier_name': 'Supplier A',
            'contact_person': 'Alice',
            'phone': '027-91030001',
            'settlement_cycle': 'monthly',
        })[0] is True
        assert_failed(create_supplier({'supplier_code': 'S9103', 'supplier_name': 'Duplicate'}))

        assert create_supplier({
            'supplier_code': 'S9104',
            'supplier_name': 'Supplier B',
            'settlement_cycle': 'quarterly',
        })[0] is True
        first = get_suppliers(search='S9103')['items'][0]

        assert_failed(update_supplier(9999, {'supplier_name': 'Missing'}))
        assert_failed(update_supplier(first['supplier_id'], {'supplier_code': ''}))
        assert_failed(update_supplier(first['supplier_id'], {'supplier_code': 'S9104'}))
        assert_failed(update_supplier(first['supplier_id'], {'settlement_cycle': 'yearly'}))
        assert_failed(update_supplier(first['supplier_id'], {'status': 'paused'}))

        assert update_supplier(first['supplier_id'], {
            'supplier_code': 'S9105',
            'supplier_name': 'Supplier A Updated',
            'contact_person': 'Bob',
            'phone': '027-91050001',
            'settlement_cycle': 'weekly',
        })[0] is True
        updated = get_suppliers(search='S9105')['items'][0]
        assert updated['supplier_name'] == 'Supplier A Updated'
        assert updated['contact_person'] == 'Bob'
        assert updated['settlement_cycle'] == 'weekly'

        assert_failed(upsert_system_setting({'setting_key': '', 'setting_value': 'value'}))
        assert_failed(upsert_system_setting({'setting_key': 'empty.value', 'setting_value': ''}))
        assert_failed(upsert_system_setting({'setting_key': 123, 'setting_value': 'value'}))
        assert upsert_system_setting({
            'setting_key': 'existing.key',
            'setting_value': 'first',
            'description': 'created',
        })[0] is True
        assert upsert_system_setting({
            'setting_key': 'existing.key',
            'setting_value': 'second',
            'description': 'updated',
        })[0] is True

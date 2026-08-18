from flask import jsonify, render_template, request

from app.routes.common import admin_required
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


PAGE_CONFIGS = {
    'members': {
        'title': '会员管理',
        'active_page': 'members',
        'api_base': '/api/members',
        'id_field': 'member_id',
        'search_placeholder': '搜索会员编号、姓名、手机号...',
        'fields': [
            {'name': 'member_no', 'label': '会员编号', 'required': True},
            {'name': 'member_name', 'label': '姓名', 'required': True},
            {'name': 'phone', 'label': '手机号'},
            {'name': 'level', 'label': '等级', 'type': 'select', 'options': [
                {'value': 'normal', 'label': '普通'},
                {'value': 'silver', 'label': '银卡'},
                {'value': 'gold', 'label': '金卡'},
                {'value': 'vip', 'label': 'VIP'},
            ]},
            {'name': 'points', 'label': '积分', 'type': 'number'},
        ],
        'columns': [
            {'name': 'member_no', 'label': '会员编号'},
            {'name': 'member_name', 'label': '姓名'},
            {'name': 'phone', 'label': '手机号'},
            {'name': 'level', 'label': '等级'},
            {'name': 'points', 'label': '积分'},
            {'name': 'status', 'label': '状态'},
        ],
        'extra_action': 'points',
    },
    'employees': {
        'title': '员工管理',
        'active_page': 'employees',
        'api_base': '/api/employees',
        'id_field': 'employee_id',
        'search_placeholder': '搜索员工编号、姓名、岗位、手机号...',
        'fields': [
            {'name': 'employee_no', 'label': '员工编号', 'required': True},
            {'name': 'employee_name', 'label': '姓名', 'required': True},
            {'name': 'position', 'label': '岗位', 'required': True},
            {'name': 'phone', 'label': '手机号'},
            {'name': 'work_schedule', 'label': '排班'},
        ],
        'columns': [
            {'name': 'employee_no', 'label': '员工编号'},
            {'name': 'employee_name', 'label': '姓名'},
            {'name': 'position', 'label': '岗位'},
            {'name': 'phone', 'label': '手机号'},
            {'name': 'work_schedule', 'label': '排班'},
            {'name': 'status', 'label': '状态'},
        ],
    },
    'suppliers': {
        'title': '供应商管理',
        'active_page': 'suppliers',
        'api_base': '/api/suppliers',
        'id_field': 'supplier_id',
        'search_placeholder': '搜索供应商编码、名称、联系人...',
        'fields': [
            {'name': 'supplier_code', 'label': '供应商编码', 'required': True},
            {'name': 'supplier_name', 'label': '供应商名称', 'required': True},
            {'name': 'contact_person', 'label': '联系人'},
            {'name': 'phone', 'label': '联系电话'},
            {'name': 'settlement_cycle', 'label': '结算周期', 'type': 'select', 'options': [
                {'value': 'weekly', 'label': '周结'},
                {'value': 'monthly', 'label': '月结'},
                {'value': 'quarterly', 'label': '季结'},
            ]},
        ],
        'columns': [
            {'name': 'supplier_code', 'label': '供应商编码'},
            {'name': 'supplier_name', 'label': '供应商名称'},
            {'name': 'contact_person', 'label': '联系人'},
            {'name': 'phone', 'label': '联系电话'},
            {'name': 'settlement_cycle', 'label': '结算周期'},
            {'name': 'status', 'label': '状态'},
        ],
    },
    'system_settings': {
        'title': '系统管理',
        'active_page': 'system_settings',
        'api_base': '/api/system-settings',
        'id_field': 'setting_id',
        'search_placeholder': '搜索参数键、参数值、说明...',
        'fields': [
            {'name': 'setting_key', 'label': '参数键', 'required': True},
            {'name': 'setting_value', 'label': '参数值', 'required': True},
            {'name': 'description', 'label': '说明'},
        ],
        'columns': [
            {'name': 'setting_key', 'label': '参数键'},
            {'name': 'setting_value', 'label': '参数值'},
            {'name': 'description', 'label': '说明'},
            {'name': 'updated_at', 'label': '更新时间'},
        ],
        'disable_status': True,
    },
}


def _request_page_args():
    return (
        request.args.get('page', 1, type=int),
        request.args.get('per_page', 20, type=int),
        request.args.get('search', ''),
        request.args.get('status'),
    )


def register_routes(app):
    @app.route('/members')
    @admin_required
    def members_page():
        return render_template('master_data.html', config=PAGE_CONFIGS['members'])

    @app.route('/employees')
    @admin_required
    def employees_page():
        return render_template('master_data.html', config=PAGE_CONFIGS['employees'])

    @app.route('/suppliers')
    @admin_required
    def suppliers_page():
        return render_template('master_data.html', config=PAGE_CONFIGS['suppliers'])

    @app.route('/system-settings')
    @admin_required
    def system_settings_page():
        return render_template('master_data.html', config=PAGE_CONFIGS['system_settings'])

    @app.route('/api/members', methods=['GET'])
    @admin_required
    def api_members():
        page, per_page, search, status = _request_page_args()
        return jsonify(get_members(page, per_page, search, status))

    @app.route('/api/members', methods=['POST'])
    @admin_required
    def api_create_member():
        success, message = create_member(request.get_json() or {})
        return jsonify({'success': success, 'message': message})

    @app.route('/api/members/<int:member_id>', methods=['PUT'])
    @admin_required
    def api_update_member(member_id):
        success, message = update_member(member_id, request.get_json() or {})
        return jsonify({'success': success, 'message': message})

    @app.route('/api/members/<int:member_id>/status', methods=['POST'])
    @admin_required
    def api_member_status(member_id):
        success, message = set_member_status(member_id, (request.get_json() or {}).get('status'))
        return jsonify({'success': success, 'message': message})

    @app.route('/api/members/<int:member_id>/points', methods=['POST'])
    @admin_required
    def api_member_points(member_id):
        data = request.get_json() or {}
        success, message = adjust_member_points(member_id, data.get('points_delta'), data.get('reason', ''))
        return jsonify({'success': success, 'message': message})

    @app.route('/api/employees', methods=['GET'])
    @admin_required
    def api_employees():
        page, per_page, search, status = _request_page_args()
        return jsonify(get_employees(page, per_page, search, status))

    @app.route('/api/employees', methods=['POST'])
    @admin_required
    def api_create_employee():
        success, message = create_employee(request.get_json() or {})
        return jsonify({'success': success, 'message': message})

    @app.route('/api/employees/<int:employee_id>', methods=['PUT'])
    @admin_required
    def api_update_employee(employee_id):
        success, message = update_employee(employee_id, request.get_json() or {})
        return jsonify({'success': success, 'message': message})

    @app.route('/api/employees/<int:employee_id>/status', methods=['POST'])
    @admin_required
    def api_employee_status(employee_id):
        success, message = set_employee_status(employee_id, (request.get_json() or {}).get('status'))
        return jsonify({'success': success, 'message': message})

    @app.route('/api/suppliers', methods=['GET'])
    @admin_required
    def api_suppliers():
        page, per_page, search, status = _request_page_args()
        return jsonify(get_suppliers(page, per_page, search, status))

    @app.route('/api/suppliers', methods=['POST'])
    @admin_required
    def api_create_supplier():
        success, message = create_supplier(request.get_json() or {})
        return jsonify({'success': success, 'message': message})

    @app.route('/api/suppliers/<int:supplier_id>', methods=['PUT'])
    @admin_required
    def api_update_supplier(supplier_id):
        success, message = update_supplier(supplier_id, request.get_json() or {})
        return jsonify({'success': success, 'message': message})

    @app.route('/api/suppliers/<int:supplier_id>/status', methods=['POST'])
    @admin_required
    def api_supplier_status(supplier_id):
        success, message = set_supplier_status(supplier_id, (request.get_json() or {}).get('status'))
        return jsonify({'success': success, 'message': message})

    @app.route('/api/system-settings', methods=['GET'])
    @admin_required
    def api_system_settings():
        page, per_page, search, _status = _request_page_args()
        return jsonify(get_system_settings(page, per_page, search))

    @app.route('/api/system-settings', methods=['POST'])
    @admin_required
    def api_upsert_system_setting():
        success, message = upsert_system_setting(request.get_json() or {})
        return jsonify({'success': success, 'message': message})

    @app.route('/api/system-settings/<setting_key>', methods=['PUT'])
    @admin_required
    def api_update_system_setting(setting_key):
        data = request.get_json() or {}
        data['setting_key'] = setting_key
        success, message = upsert_system_setting(data)
        return jsonify({'success': success, 'message': message})


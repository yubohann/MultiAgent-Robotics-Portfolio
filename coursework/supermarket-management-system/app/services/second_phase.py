from datetime import datetime

from app import db
from app.models import Employee, Member, Supplier, SystemSetting


ACTIVE_STATUSES = {'active', 'inactive'}
MEMBER_LEVELS = {'normal', 'silver', 'gold', 'vip'}
SETTLEMENT_CYCLES = {'weekly', 'monthly', 'quarterly'}


def _clean(value):
    return (value or '').strip()


def _paginate(query, page, per_page):
    return query.paginate(page=page, per_page=per_page, error_out=False)


def _format_datetime(value):
    return value.strftime('%Y-%m-%d') if value else '-'


def _validate_status(status):
    status = _clean(status) or 'active'
    if status not in ACTIVE_STATUSES:
        raise ValueError('状态必须为 active 或 inactive')
    return status


def get_members(page=1, per_page=20, search='', status=None):
    query = Member.query
    search = _clean(search)
    if search:
        query = query.filter(db.or_(
            Member.member_no.like(f'%{search}%'),
            Member.member_name.like(f'%{search}%'),
            Member.phone.like(f'%{search}%'),
        ))
    if status in ACTIVE_STATUSES:
        query = query.filter_by(status=status)

    pagination = _paginate(query.order_by(Member.member_id.desc()), page, per_page)
    return {
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page,
        'items': [
            {
                'member_id': item.member_id,
                'member_no': item.member_no,
                'member_name': item.member_name,
                'phone': item.phone or '',
                'level': item.level,
                'points': item.points,
                'status': item.status,
                'registered_at': _format_datetime(item.registered_at),
            }
            for item in pagination.items
        ],
    }


def create_member(data):
    try:
        member_no = _clean(data.get('member_no'))
        member_name = _clean(data.get('member_name'))
        phone = _clean(data.get('phone')) or None
        level = _clean(data.get('level')) or 'normal'

        if not member_no or not member_name:
            return False, '会员编号和姓名不能为空'
        if level not in MEMBER_LEVELS:
            return False, '会员等级无效'
        if Member.query.filter_by(member_no=member_no).first():
            return False, '会员编号已存在'
        if phone and Member.query.filter_by(phone=phone).first():
            return False, '手机号已存在'

        member = Member(
            member_no=member_no,
            member_name=member_name,
            phone=phone,
            level=level,
            points=max(int(data.get('points') or 0), 0),
            status=_validate_status(data.get('status')),
        )
        db.session.add(member)
        db.session.commit()
        return True, '会员创建成功'
    except Exception as exc:
        db.session.rollback()
        return False, f'会员创建失败：{exc}'


def update_member(member_id, data):
    try:
        member = db.session.get(Member, member_id)
        if not member:
            return False, '会员不存在'

        if 'member_no' in data:
            member_no = _clean(data.get('member_no'))
            if not member_no:
                return False, '会员编号不能为空'
            existing = Member.query.filter_by(member_no=member_no).first()
            if existing and existing.member_id != member_id:
                return False, '会员编号已存在'
            member.member_no = member_no
        if 'member_name' in data:
            member.member_name = _clean(data.get('member_name'))
        if 'phone' in data:
            phone = _clean(data.get('phone')) or None
            existing = Member.query.filter_by(phone=phone).first() if phone else None
            if existing and existing.member_id != member_id:
                return False, '手机号已存在'
            member.phone = phone
        if 'level' in data:
            level = _clean(data.get('level')) or 'normal'
            if level not in MEMBER_LEVELS:
                return False, '会员等级无效'
            member.level = level
        if 'points' in data:
            member.points = max(int(data.get('points') or 0), 0)
        if 'status' in data:
            member.status = _validate_status(data.get('status'))

        member.updated_at = datetime.now()
        db.session.commit()
        return True, '会员更新成功'
    except Exception as exc:
        db.session.rollback()
        return False, f'会员更新失败：{exc}'


def adjust_member_points(member_id, points_delta, reason=''):
    try:
        member = db.session.get(Member, member_id)
        if not member:
            return False, '会员不存在'
        delta = int(points_delta or 0)
        if member.points + delta < 0:
            return False, '积分不足'
        member.points += delta
        member.updated_at = datetime.now()
        db.session.commit()
        suffix = f'，原因：{_clean(reason)}' if _clean(reason) else ''
        return True, f'会员积分已调整{suffix}'
    except Exception as exc:
        db.session.rollback()
        return False, f'积分调整失败：{exc}'


def set_member_status(member_id, status):
    return update_member(member_id, {'status': status})


def get_employees(page=1, per_page=20, search='', status=None):
    query = Employee.query
    search = _clean(search)
    if search:
        query = query.filter(db.or_(
            Employee.employee_no.like(f'%{search}%'),
            Employee.employee_name.like(f'%{search}%'),
            Employee.position.like(f'%{search}%'),
            Employee.phone.like(f'%{search}%'),
        ))
    if status in ACTIVE_STATUSES:
        query = query.filter_by(status=status)

    pagination = _paginate(query.order_by(Employee.employee_id.desc()), page, per_page)
    return {
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page,
        'items': [
            {
                'employee_id': item.employee_id,
                'employee_no': item.employee_no,
                'employee_name': item.employee_name,
                'position': item.position,
                'phone': item.phone or '',
                'work_schedule': item.work_schedule or '',
                'status': item.status,
                'hired_at': _format_datetime(item.hired_at),
            }
            for item in pagination.items
        ],
    }


def create_employee(data):
    try:
        employee_no = _clean(data.get('employee_no'))
        employee_name = _clean(data.get('employee_name'))
        position = _clean(data.get('position'))
        phone = _clean(data.get('phone')) or None

        if not employee_no or not employee_name or not position:
            return False, '员工编号、姓名和岗位不能为空'
        if Employee.query.filter_by(employee_no=employee_no).first():
            return False, '员工编号已存在'
        if phone and Employee.query.filter_by(phone=phone).first():
            return False, '手机号已存在'

        db.session.add(Employee(
            employee_no=employee_no,
            employee_name=employee_name,
            position=position,
            phone=phone,
            work_schedule=_clean(data.get('work_schedule')),
            status=_validate_status(data.get('status')),
        ))
        db.session.commit()
        return True, '员工档案创建成功'
    except Exception as exc:
        db.session.rollback()
        return False, f'员工创建失败：{exc}'


def update_employee(employee_id, data):
    try:
        employee = db.session.get(Employee, employee_id)
        if not employee:
            return False, '员工不存在'

        if 'employee_no' in data:
            employee_no = _clean(data.get('employee_no'))
            if not employee_no:
                return False, '员工编号不能为空'
            existing = Employee.query.filter_by(employee_no=employee_no).first()
            if existing and existing.employee_id != employee_id:
                return False, '员工编号已存在'
            employee.employee_no = employee_no
        if 'employee_name' in data:
            employee.employee_name = _clean(data.get('employee_name'))
        if 'position' in data:
            employee.position = _clean(data.get('position'))
        if 'phone' in data:
            phone = _clean(data.get('phone')) or None
            existing = Employee.query.filter_by(phone=phone).first() if phone else None
            if existing and existing.employee_id != employee_id:
                return False, '手机号已存在'
            employee.phone = phone
        if 'work_schedule' in data:
            employee.work_schedule = _clean(data.get('work_schedule'))
        if 'status' in data:
            employee.status = _validate_status(data.get('status'))

        employee.updated_at = datetime.now()
        db.session.commit()
        return True, '员工档案更新成功'
    except Exception as exc:
        db.session.rollback()
        return False, f'员工更新失败：{exc}'


def set_employee_status(employee_id, status):
    return update_employee(employee_id, {'status': status})


def get_suppliers(page=1, per_page=20, search='', status=None):
    query = Supplier.query
    search = _clean(search)
    if search:
        query = query.filter(db.or_(
            Supplier.supplier_code.like(f'%{search}%'),
            Supplier.supplier_name.like(f'%{search}%'),
            Supplier.contact_person.like(f'%{search}%'),
            Supplier.phone.like(f'%{search}%'),
        ))
    if status in ACTIVE_STATUSES:
        query = query.filter_by(status=status)

    pagination = _paginate(query.order_by(Supplier.supplier_id.desc()), page, per_page)
    return {
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page,
        'items': [
            {
                'supplier_id': item.supplier_id,
                'supplier_code': item.supplier_code,
                'supplier_name': item.supplier_name,
                'contact_person': item.contact_person or '',
                'phone': item.phone or '',
                'settlement_cycle': item.settlement_cycle,
                'status': item.status,
                'created_at': _format_datetime(item.created_at),
            }
            for item in pagination.items
        ],
    }


def create_supplier(data):
    try:
        supplier_code = _clean(data.get('supplier_code'))
        supplier_name = _clean(data.get('supplier_name'))
        cycle = _clean(data.get('settlement_cycle')) or 'monthly'

        if not supplier_code or not supplier_name:
            return False, '供应商编码和名称不能为空'
        if cycle not in SETTLEMENT_CYCLES:
            return False, '结算周期无效'
        if Supplier.query.filter_by(supplier_code=supplier_code).first():
            return False, '供应商编码已存在'

        db.session.add(Supplier(
            supplier_code=supplier_code,
            supplier_name=supplier_name,
            contact_person=_clean(data.get('contact_person')),
            phone=_clean(data.get('phone')),
            settlement_cycle=cycle,
            status=_validate_status(data.get('status')),
        ))
        db.session.commit()
        return True, '供应商创建成功'
    except Exception as exc:
        db.session.rollback()
        return False, f'供应商创建失败：{exc}'


def update_supplier(supplier_id, data):
    try:
        supplier = db.session.get(Supplier, supplier_id)
        if not supplier:
            return False, '供应商不存在'

        if 'supplier_code' in data:
            supplier_code = _clean(data.get('supplier_code'))
            if not supplier_code:
                return False, '供应商编码不能为空'
            existing = Supplier.query.filter_by(supplier_code=supplier_code).first()
            if existing and existing.supplier_id != supplier_id:
                return False, '供应商编码已存在'
            supplier.supplier_code = supplier_code
        if 'supplier_name' in data:
            supplier.supplier_name = _clean(data.get('supplier_name'))
        if 'contact_person' in data:
            supplier.contact_person = _clean(data.get('contact_person'))
        if 'phone' in data:
            supplier.phone = _clean(data.get('phone'))
        if 'settlement_cycle' in data:
            cycle = _clean(data.get('settlement_cycle')) or 'monthly'
            if cycle not in SETTLEMENT_CYCLES:
                return False, '结算周期无效'
            supplier.settlement_cycle = cycle
        if 'status' in data:
            supplier.status = _validate_status(data.get('status'))

        supplier.updated_at = datetime.now()
        db.session.commit()
        return True, '供应商更新成功'
    except Exception as exc:
        db.session.rollback()
        return False, f'供应商更新失败：{exc}'


def set_supplier_status(supplier_id, status):
    return update_supplier(supplier_id, {'status': status})


def get_system_settings(page=1, per_page=20, search=''):
    query = SystemSetting.query
    search = _clean(search)
    if search:
        query = query.filter(db.or_(
            SystemSetting.setting_key.like(f'%{search}%'),
            SystemSetting.setting_value.like(f'%{search}%'),
            SystemSetting.description.like(f'%{search}%'),
        ))

    pagination = _paginate(query.order_by(SystemSetting.setting_key.asc()), page, per_page)
    return {
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page,
        'items': [
            {
                'setting_id': item.setting_id,
                'setting_key': item.setting_key,
                'setting_value': item.setting_value,
                'description': item.description or '',
                'updated_at': _format_datetime(item.updated_at),
            }
            for item in pagination.items
        ],
    }


def upsert_system_setting(data):
    try:
        setting_key = _clean(data.get('setting_key'))
        setting_value = _clean(data.get('setting_value'))
        if not setting_key:
            return False, '参数键不能为空'
        if setting_value == '':
            return False, '参数值不能为空'

        setting = SystemSetting.query.filter_by(setting_key=setting_key).first()
        if setting:
            setting.setting_value = setting_value
            setting.description = _clean(data.get('description'))
            setting.updated_at = datetime.now()
            message = '系统参数更新成功'
        else:
            setting = SystemSetting(
                setting_key=setting_key,
                setting_value=setting_value,
                description=_clean(data.get('description')),
            )
            db.session.add(setting)
            message = '系统参数创建成功'

        db.session.commit()
        return True, message
    except Exception as exc:
        db.session.rollback()
        return False, f'系统参数保存失败：{exc}'


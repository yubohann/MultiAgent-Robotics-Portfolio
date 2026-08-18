from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
import random
import re

from sqlalchemy import func

from app import db
from app.models import CashReconciliation, FinancePeriodClose, FinanceTransaction, PayablePayment, Sale, SaleItem, SupplierPayable, User

PAYMENT_METHOD_LABELS = {
    'cash': '现金',
    'wechat': '微信',
    'alipay': '支付宝',
    'card': '银行卡',
    'bank_transfer': '银行转账',
}

TRANSACTION_TYPE_LABELS = {
    'income': '收入',
    'expense': '支出',
}

PAYABLE_STATUS_LABELS = {
    'unpaid': '未支付',
    'partial': '部分支付',
    'paid': '已支付',
    'overdue': '已逾期',
}

DECIMAL_ZERO = Decimal('0.00')


def _to_decimal(value):
    try:
        return Decimal(str(value or 0)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    except Exception:
        return DECIMAL_ZERO


def _round_float(value):
    return round(float(value or 0), 2)


def _parse_date(value):
    if not value:
        return None

    try:
        return datetime.strptime(str(value), '%Y-%m-%d').date()
    except ValueError:
        return None


def _parse_datetime(value):
    if not value:
        return datetime.now()

    text = str(value).strip()
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d'):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return datetime.now()


def _month_range(period_month):
    start = datetime.strptime(f'{period_month}-01', '%Y-%m-%d').date()
    if start.month == 12:
        end = date(start.year + 1, 1, 1) - timedelta(days=1)
    else:
        end = date(start.year, start.month + 1, 1) - timedelta(days=1)
    return start, end


def _compute_payable_status(total_amount, paid_amount, due_date):
    total = _to_decimal(total_amount)
    paid = _to_decimal(paid_amount)
    remaining = total - paid

    if remaining <= 0:
        return 'paid'
    if paid > 0:
        if due_date and due_date < date.today():
            return 'overdue'
        return 'partial'
    if due_date and due_date < date.today():
        return 'overdue'
    return 'unpaid'


def _generate_transaction_no():
    now = datetime.now().strftime('%Y%m%d%H%M%S')
    return f'FT{now}{random.randint(1000, 9999)}'


def _sales_amount_between(start_date, end_date):
    query = db.session.query(func.coalesce(func.sum(Sale.actual_amount), 0)).filter(
        Sale.status == 'completed',
        db.func.date(Sale.created_at) >= start_date,
        db.func.date(Sale.created_at) <= end_date,
    )
    result = query.scalar() or 0
    return _to_decimal(result)


def _gross_profit_between(start_date, end_date):
    query = db.session.query(
        func.coalesce(func.sum((SaleItem.unit_price - SaleItem.purchase_price) * SaleItem.quantity), 0)
    ).join(Sale).filter(
        Sale.status == 'completed',
        db.func.date(Sale.created_at) >= start_date,
        db.func.date(Sale.created_at) <= end_date,
    )
    result = query.scalar() or 0
    return _to_decimal(result)


def _other_income_between(start_date, end_date):
    result = db.session.query(func.coalesce(func.sum(FinanceTransaction.amount), 0)).filter(
        FinanceTransaction.transaction_type == 'income',
        db.func.date(FinanceTransaction.occurred_at) >= start_date,
        db.func.date(FinanceTransaction.occurred_at) <= end_date,
    ).scalar() or 0
    return _to_decimal(result)


def _expense_between(start_date, end_date):
    result = db.session.query(func.coalesce(func.sum(FinanceTransaction.amount), 0)).filter(
        FinanceTransaction.transaction_type == 'expense',
        db.func.date(FinanceTransaction.occurred_at) >= start_date,
        db.func.date(FinanceTransaction.occurred_at) <= end_date,
    ).scalar() or 0
    return _to_decimal(result)


def get_finance_overview(period='month'):
    now = datetime.now().date()
    if period == 'today':
        start_date = now
    elif period == 'week':
        start_date = now - timedelta(days=6)
    else:
        start_date = now.replace(day=1)
    end_date = now

    sales_income = _sales_amount_between(start_date, end_date)
    other_income = _other_income_between(start_date, end_date)
    expense_amount = _expense_between(start_date, end_date)
    gross_profit = _gross_profit_between(start_date, end_date)

    cash_inflow = sales_income + other_income
    cash_outflow = expense_amount
    net_profit = gross_profit + other_income - expense_amount

    pending_payables = db.session.query(
        func.coalesce(func.sum(SupplierPayable.total_amount - SupplierPayable.paid_amount), 0)
    ).filter(
        SupplierPayable.total_amount > SupplierPayable.paid_amount
    ).scalar() or 0

    today_expected_methods = db.session.query(Sale.payment_method).filter(
        Sale.status == 'completed',
        db.func.date(Sale.created_at) == str(now),
        Sale.payment_method.isnot(None),
    ).distinct().all()
    expected_method_count = len([row[0] for row in today_expected_methods if row[0]])
    reconciled_method_count = CashReconciliation.query.filter_by(reconcile_date=now).count()
    unreconciled_count = max(expected_method_count - reconciled_method_count, 0)

    return {
        'sales_income': _round_float(sales_income),
        'other_income': _round_float(other_income),
        'expense_amount': _round_float(expense_amount),
        'gross_profit': _round_float(gross_profit),
        'net_profit': _round_float(net_profit),
        'cash_inflow': _round_float(cash_inflow),
        'cash_outflow': _round_float(cash_outflow),
        'pending_payables': _round_float(pending_payables),
        'unreconciled_count': int(unreconciled_count),
    }


def get_finance_transactions(page=1, per_page=10, transaction_type=None, category='', search='', start_date=None, end_date=None):
    query = db.session.query(
        FinanceTransaction,
        User.username.label('operator_username'),
        User.real_name.label('operator_real_name'),
    ).outerjoin(User, FinanceTransaction.operator_id == User.user_id)

    if transaction_type:
        query = query.filter(FinanceTransaction.transaction_type == transaction_type)

    if category:
        query = query.filter(FinanceTransaction.category.like(f'%{category}%'))

    if search:
        key = f'%{search}%'
        query = query.filter(
            db.or_(
                FinanceTransaction.transaction_no.like(key),
                FinanceTransaction.category.like(key),
                FinanceTransaction.description.like(key),
                FinanceTransaction.related_order_no.like(key),
                User.username.like(key),
                User.real_name.like(key),
            )
        )

    if start_date:
        query = query.filter(db.func.date(FinanceTransaction.occurred_at) >= start_date)

    if end_date:
        query = query.filter(db.func.date(FinanceTransaction.occurred_at) <= end_date)

    query = query.order_by(FinanceTransaction.occurred_at.desc(), FinanceTransaction.transaction_id.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    items = []
    for row, operator_username, operator_real_name in pagination.items:
        operator_name = operator_real_name or operator_username or '-'
        items.append({
            'transaction_id': row.transaction_id,
            'transaction_no': row.transaction_no,
            'transaction_type': row.transaction_type,
            'transaction_type_label': TRANSACTION_TYPE_LABELS.get(row.transaction_type, row.transaction_type),
            'category': row.category,
            'amount': _round_float(row.amount),
            'payment_method': row.payment_method or '',
            'payment_method_label': PAYMENT_METHOD_LABELS.get(row.payment_method, row.payment_method or '-'),
            'related_order_no': row.related_order_no or '-',
            'description': row.description or '-',
            'operator_name': operator_name,
            'occurred_at': row.occurred_at.strftime('%Y-%m-%d %H:%M:%S') if row.occurred_at else '-',
        })

    return {
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page,
        'items': items,
    }


def create_finance_transaction(transaction_type, category, amount, payment_method=None, occurred_at=None, description=None, related_order_no=None, operator_id=None):
    if transaction_type not in TRANSACTION_TYPE_LABELS:
        return False, '流水类型不合法'

    category = str(category or '').strip()
    if not category:
        return False, '科目不能为空'

    amount_decimal = _to_decimal(amount)
    if amount_decimal <= 0:
        return False, '金额必须大于 0'

    if payment_method and payment_method not in PAYMENT_METHOD_LABELS:
        return False, '支付方式不支持'

    transaction = FinanceTransaction(
        transaction_no=_generate_transaction_no(),
        transaction_type=transaction_type,
        category=category,
        amount=amount_decimal,
        payment_method=(payment_method or None),
        related_order_no=(str(related_order_no).strip() or None) if related_order_no is not None else None,
        description=(str(description).strip() or None) if description is not None else None,
        occurred_at=_parse_datetime(occurred_at),
        operator_id=operator_id,
    )

    try:
        db.session.add(transaction)
        db.session.commit()
        return True, '流水创建成功'
    except Exception as exc:
        db.session.rollback()
        return False, f'流水创建失败：{str(exc)}'


def get_reconciliation(date_text=None):
    target_date = _parse_date(date_text) or date.today()

    expected_rows = db.session.query(
        Sale.payment_method,
        func.coalesce(func.sum(Sale.actual_amount), 0),
        func.count(Sale.sale_id),
    ).filter(
        Sale.status == 'completed',
        db.func.date(Sale.created_at) == str(target_date),
        Sale.payment_method.isnot(None),
    ).group_by(Sale.payment_method).all()

    expected_map = {
        method: {
            'expected_amount': _to_decimal(amount),
            'order_count': int(order_count or 0),
        }
        for method, amount, order_count in expected_rows
        if method
    }

    records = CashReconciliation.query.filter_by(reconcile_date=target_date).all()
    record_map = {item.payment_method: item for item in records}

    methods = sorted(set(expected_map.keys()) | set(record_map.keys()))
    items = []
    expected_total = DECIMAL_ZERO
    actual_total = DECIMAL_ZERO

    for method in methods:
        expected_amount = expected_map.get(method, {}).get('expected_amount', DECIMAL_ZERO)
        order_count = expected_map.get(method, {}).get('order_count', 0)

        record = record_map.get(method)
        if record:
            actual_amount = _to_decimal(record.actual_amount)
            difference_amount = _to_decimal(record.difference_amount)
            note = record.note or ''
            reconciliation_id = record.reconciliation_id
        else:
            actual_amount = expected_amount
            difference_amount = DECIMAL_ZERO
            note = ''
            reconciliation_id = None

        status = 'matched'
        if difference_amount > 0:
            status = 'over'
        elif difference_amount < 0:
            status = 'short'

        expected_total += expected_amount
        actual_total += actual_amount

        items.append({
            'reconciliation_id': reconciliation_id,
            'payment_method': method,
            'payment_method_label': PAYMENT_METHOD_LABELS.get(method, method),
            'order_count': order_count,
            'expected_amount': _round_float(expected_amount),
            'actual_amount': _round_float(actual_amount),
            'difference_amount': _round_float(difference_amount),
            'status': status,
            'note': note,
        })

    diff_total = actual_total - expected_total
    mismatch_count = len([item for item in items if item['status'] != 'matched'])

    return {
        'date': target_date.strftime('%Y-%m-%d'),
        'items': items,
        'summary': {
            'expected_total': _round_float(expected_total),
            'actual_total': _round_float(actual_total),
            'difference_total': _round_float(diff_total),
            'mismatch_count': mismatch_count,
        },
    }


def save_reconciliation(date_text, payment_method, actual_amount, note='', operator_id=None):
    target_date = _parse_date(date_text)
    if not target_date:
        return False, '对账日期不合法'

    if payment_method not in PAYMENT_METHOD_LABELS:
        return False, '支付方式不合法'

    actual = _to_decimal(actual_amount)
    if actual < 0:
        return False, '实收金额不能小于 0'

    expected = db.session.query(func.coalesce(func.sum(Sale.actual_amount), 0)).filter(
        Sale.status == 'completed',
        Sale.payment_method == payment_method,
        db.func.date(Sale.created_at) == str(target_date),
    ).scalar() or 0
    expected = _to_decimal(expected)
    difference = actual - expected

    record = CashReconciliation.query.filter_by(
        reconcile_date=target_date,
        payment_method=payment_method,
    ).first()

    if not record:
        record = CashReconciliation(
            reconcile_date=target_date,
            payment_method=payment_method,
            created_by=operator_id,
        )
        db.session.add(record)

    record.expected_amount = expected
    record.actual_amount = actual
    record.difference_amount = difference
    record.note = (str(note).strip() or None) if note is not None else None

    try:
        db.session.commit()
        return True, '对账保存成功'
    except Exception as exc:
        db.session.rollback()
        return False, f'对账保存失败：{str(exc)}'


def get_payables(page=1, per_page=10, status=None, search='', due_start=None, due_end=None):
    query = SupplierPayable.query
    today = date.today()

    if search:
        key = f'%{search}%'
        query = query.filter(
            db.or_(
                SupplierPayable.supplier_name.like(key),
                SupplierPayable.bill_no.like(key),
                SupplierPayable.note.like(key),
            )
        )

    due_start_date = _parse_date(due_start)
    due_end_date = _parse_date(due_end)

    if due_start_date:
        query = query.filter(SupplierPayable.due_date >= due_start_date)
    if due_end_date:
        query = query.filter(SupplierPayable.due_date <= due_end_date)

    if status == 'paid':
        query = query.filter(SupplierPayable.total_amount <= SupplierPayable.paid_amount)
    elif status == 'overdue':
        query = query.filter(
            SupplierPayable.due_date < today,
            SupplierPayable.total_amount > SupplierPayable.paid_amount,
        )
    elif status == 'partial':
        query = query.filter(
            SupplierPayable.paid_amount > 0,
            SupplierPayable.total_amount > SupplierPayable.paid_amount,
            SupplierPayable.due_date >= today,
        )
    elif status == 'unpaid':
        query = query.filter(
            SupplierPayable.paid_amount <= 0,
            SupplierPayable.total_amount > 0,
            SupplierPayable.due_date >= today,
        )

    query = query.order_by(SupplierPayable.due_date.asc(), SupplierPayable.payable_id.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    items = []
    for row in pagination.items:
        total = _to_decimal(row.total_amount)
        paid = _to_decimal(row.paid_amount)
        remaining = total - paid
        runtime_status = _compute_payable_status(total, paid, row.due_date)

        items.append({
            'payable_id': row.payable_id,
            'supplier_name': row.supplier_name,
            'bill_no': row.bill_no or '-',
            'total_amount': _round_float(total),
            'paid_amount': _round_float(paid),
            'remaining_amount': _round_float(remaining),
            'due_date': row.due_date.strftime('%Y-%m-%d') if row.due_date else '-',
            'status': runtime_status,
            'status_label': PAYABLE_STATUS_LABELS.get(runtime_status, runtime_status),
            'note': row.note or '-',
            'created_at': row.created_at.strftime('%Y-%m-%d %H:%M:%S') if row.created_at else '-',
        })

    return {
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page,
        'items': items,
    }


def create_payable(supplier_name, total_amount, due_date, bill_no=None, note=None, created_by=None):
    supplier_name = str(supplier_name or '').strip()
    if not supplier_name:
        return False, '供应商名称不能为空'

    due_date_obj = _parse_date(due_date)
    if not due_date_obj:
        return False, '到期日格式不正确'

    amount = _to_decimal(total_amount)
    if amount <= 0:
        return False, '应付金额必须大于 0'

    bill_no = str(bill_no or '').strip() or None
    if bill_no and SupplierPayable.query.filter_by(bill_no=bill_no).first():
        return False, '账单号已存在'

    payable = SupplierPayable(
        supplier_name=supplier_name,
        bill_no=bill_no,
        total_amount=amount,
        paid_amount=DECIMAL_ZERO,
        due_date=due_date_obj,
        status=_compute_payable_status(amount, DECIMAL_ZERO, due_date_obj),
        note=(str(note).strip() or None) if note is not None else None,
        created_by=created_by,
    )

    try:
        db.session.add(payable)
        db.session.commit()
        return True, '应付款创建成功'
    except Exception as exc:
        db.session.rollback()
        return False, f'应付款创建失败：{str(exc)}'


def record_payable_payment(payable_id, amount, payment_method='bank_transfer', paid_at=None, remark=None, operator_id=None):
    payable = SupplierPayable.query.get(payable_id)
    if not payable:
        return False, '应付款不存在'

    payment_amount = _to_decimal(amount)
    if payment_amount <= 0:
        return False, '支付金额必须大于 0'

    total = _to_decimal(payable.total_amount)
    paid = _to_decimal(payable.paid_amount)
    remaining = total - paid
    if payment_amount > remaining:
        return False, '支付金额不能大于剩余应付金额'

    payment_method = payment_method or 'bank_transfer'
    if payment_method not in PAYMENT_METHOD_LABELS:
        return False, '支付方式不支持'

    payment = PayablePayment(
        payable_id=payable.payable_id,
        amount=payment_amount,
        payment_method=payment_method,
        paid_at=_parse_datetime(paid_at),
        remark=(str(remark).strip() or None) if remark is not None else None,
        operator_id=operator_id,
    )

    payable.paid_amount = paid + payment_amount
    payable.status = _compute_payable_status(payable.total_amount, payable.paid_amount, payable.due_date)

    try:
        db.session.add(payment)
        db.session.commit()
        return True, '付款记录成功'
    except Exception as exc:
        db.session.rollback()
        return False, f'付款失败：{str(exc)}'


def close_finance_period(period_month, note='', operator_id=None):
    period_month = str(period_month or '').strip()
    if not re.match(r'^\d{4}-\d{2}$', period_month):
        return False, '关账月份格式错误，请使用 YYYY-MM', None

    if FinancePeriodClose.query.filter_by(period_month=period_month).first():
        return False, '该月份已完成关账', None

    try:
        start_date, end_date = _month_range(period_month)
    except ValueError:
        return False, '月份不合法', None

    sales_income = _sales_amount_between(start_date, end_date)
    other_income = _other_income_between(start_date, end_date)
    expense_amount = _expense_between(start_date, end_date)
    gross_profit = _gross_profit_between(start_date, end_date)

    cash_inflow = sales_income + other_income
    cash_outflow = expense_amount
    net_profit = gross_profit + other_income - expense_amount

    close_record = FinancePeriodClose(
        period_month=period_month,
        total_sales=sales_income,
        other_income=other_income,
        expense_amount=expense_amount,
        gross_profit=gross_profit,
        net_profit=net_profit,
        cash_inflow=cash_inflow,
        cash_outflow=cash_outflow,
        note=(str(note).strip() or None) if note is not None else None,
        closed_by=operator_id,
    )

    try:
        db.session.add(close_record)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return False, f'关账失败：{str(exc)}', None

    snapshot = {
        'period_month': period_month,
        'total_sales': _round_float(sales_income),
        'other_income': _round_float(other_income),
        'expense_amount': _round_float(expense_amount),
        'gross_profit': _round_float(gross_profit),
        'net_profit': _round_float(net_profit),
        'cash_inflow': _round_float(cash_inflow),
        'cash_outflow': _round_float(cash_outflow),
    }
    return True, '关账完成', snapshot


def get_recent_closings(limit=6):
    limit = max(1, min(int(limit or 6), 24))
    rows = FinancePeriodClose.query.order_by(FinancePeriodClose.closed_at.desc()).limit(limit).all()

    closings = []
    for row in rows:
        closings.append({
            'close_id': row.close_id,
            'period_month': row.period_month,
            'total_sales': _round_float(row.total_sales),
            'other_income': _round_float(row.other_income),
            'expense_amount': _round_float(row.expense_amount),
            'gross_profit': _round_float(row.gross_profit),
            'net_profit': _round_float(row.net_profit),
            'cash_inflow': _round_float(row.cash_inflow),
            'cash_outflow': _round_float(row.cash_outflow),
            'closed_at': row.closed_at.strftime('%Y-%m-%d %H:%M:%S') if row.closed_at else '-',
        })
    return closings

from app import db
from app.models import Sale, SaleItem, User
from sqlalchemy import func


def get_sales_orders(page=1, per_page=10, search='', cashier_id=None, status=None, payment_method=None, start_date=None, end_date=None):
    """获取销售订单列表"""
    query = db.session.query(
        Sale.sale_id,
        Sale.order_no,
        Sale.cashier_id,
        Sale.total_amount,
        Sale.discount_amount,
        Sale.actual_amount,
        Sale.payment_method,
        Sale.status,
        Sale.created_at,
        User.username.label('cashier_username'),
        User.real_name.label('cashier_real_name'),
        func.count(SaleItem.item_id).label('item_count')
    ).join(
        User, Sale.cashier_id == User.user_id
    ).outerjoin(
        SaleItem, Sale.sale_id == SaleItem.sale_id
    )

    if search:
        search_key = f'%{search}%'
        query = query.filter(
            db.or_(
                Sale.order_no.like(search_key),
                User.username.like(search_key),
                User.real_name.like(search_key)
            )
        )

    if cashier_id:
        query = query.filter(Sale.cashier_id == cashier_id)

    if status:
        query = query.filter(Sale.status == status)

    if payment_method:
        query = query.filter(Sale.payment_method == payment_method)

    if start_date:
        query = query.filter(db.func.date(Sale.created_at) >= start_date)

    if end_date:
        query = query.filter(db.func.date(Sale.created_at) <= end_date)

    query = query.group_by(
        Sale.sale_id,
        Sale.order_no,
        Sale.cashier_id,
        Sale.total_amount,
        Sale.discount_amount,
        Sale.actual_amount,
        Sale.payment_method,
        Sale.status,
        Sale.created_at,
        User.user_id,
        User.username,
        User.real_name
    ).order_by(Sale.created_at.desc())

    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    status_labels = {
        'completed': '完成',
        'refunded': '已退款',
        'cancelled': '已取消'
    }
    payment_labels = {
        'cash': '现金',
        'wechat': '微信',
        'alipay': '支付宝',
        'card': '银行卡'
    }

    items = []
    for row in pagination.items:
        cashier_name = row.cashier_real_name or row.cashier_username or '-'
        items.append({
            'sale_id': row.sale_id,
            'order_no': row.order_no,
            'cashier_id': row.cashier_id,
            'cashier_name': cashier_name,
            'total_amount': round(float(row.total_amount), 2),
            'discount_amount': round(float(row.discount_amount or 0), 2),
            'actual_amount': round(float(row.actual_amount), 2),
            'payment_method': row.payment_method or '-',
            'payment_method_label': payment_labels.get(row.payment_method, row.payment_method or '-'),
            'status': row.status,
            'status_label': status_labels.get(row.status, row.status or '-'),
            'item_count': int(row.item_count or 0),
            'created_at': row.created_at.strftime('%Y-%m-%d %H:%M:%S') if row.created_at else '-'
        })

    return {
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page,
        'items': items
    }


def get_sales_order_detail(sale_id):
    """获取销售订单详情"""
    sale = db.session.get(Sale, sale_id)
    if not sale:
        return None

    payment_labels = {
        'cash': '现金',
        'wechat': '微信',
        'alipay': '支付宝',
        'card': '银行卡'
    }
    status_labels = {
        'completed': '完成',
        'refunded': '已退款',
        'cancelled': '已取消'
    }

    if sale.cashier:
        cashier_name = sale.cashier.real_name or sale.cashier.username
    else:
        cashier_name = '-'

    items = []
    for item in sale.items:
        product_name = item.product.product_name if item.product else '-'
        product_code = item.product.product_code if item.product else '-'
        items.append({
            'product_id': item.product_id,
            'product_name': product_name,
            'product_code': product_code,
            'quantity': item.quantity,
            'unit_price': round(float(item.unit_price), 2),
            'purchase_price': round(float(item.purchase_price), 2),
            'subtotal': round(float(item.subtotal), 2)
        })

    return {
        'sale_id': sale.sale_id,
        'order_no': sale.order_no,
        'cashier_name': cashier_name,
        'cashier_id': sale.cashier_id,
        'total_amount': round(float(sale.total_amount), 2),
        'discount_amount': round(float(sale.discount_amount or 0), 2),
        'actual_amount': round(float(sale.actual_amount), 2),
        'payment_method': sale.payment_method or '-',
        'payment_method_label': payment_labels.get(sale.payment_method, sale.payment_method or '-'),
        'status': sale.status,
        'status_label': status_labels.get(sale.status, sale.status or '-'),
        'created_at': sale.created_at.strftime('%Y-%m-%d %H:%M:%S') if sale.created_at else '-',
        'items': items,
        'item_count': len(items)
    }

from sqlalchemy import func

from app import db
from app.models import Category, Inventory, InventoryLog, Product, User


def set_inventory_quantity(product_id, quantity_after, change_type='adjust', reason=None, operator_id=None):
    """设置库存数量并记录流水。"""
    inventory = db.session.get(Inventory, product_id)
    quantity_before = inventory.quantity if inventory else 0

    if not inventory:
        inventory = Inventory(product_id=product_id, quantity=0)
        db.session.add(inventory)

    quantity_after = int(quantity_after)
    inventory.quantity = quantity_after

    db.session.add(
        InventoryLog(
            product_id=product_id,
            change_type=change_type,
            quantity_change=quantity_after - quantity_before,
            quantity_before=quantity_before,
            quantity_after=quantity_after,
            reason=reason,
            operator_id=operator_id,
        )
    )

    return inventory


def get_inventory_summary():
    """获取库存概览。"""
    total_products = Product.query.count()
    total_quantity = db.session.query(func.coalesce(func.sum(Inventory.quantity), 0)).scalar() or 0
    low_stock_count = db.session.query(func.count(Product.product_id)).join(
        Inventory, Product.product_id == Inventory.product_id
    ).filter(
        Product.status == 1,
        Inventory.quantity > 0,
        Inventory.quantity <= Product.min_stock,
    ).scalar() or 0
    out_stock_count = db.session.query(func.count(Product.product_id)).join(
        Inventory, Product.product_id == Inventory.product_id
    ).filter(
        Product.status == 1,
        Inventory.quantity <= 0,
    ).scalar() or 0

    return {
        'total_products': int(total_products),
        'total_quantity': int(total_quantity),
        'low_stock_count': int(low_stock_count),
        'out_stock_count': int(out_stock_count),
    }


def get_inventory_list(page=1, per_page=20, search='', category_id=None, stock_state=None):
    """获取库存列表。"""
    query = db.session.query(
        Product,
        Inventory.quantity,
        Category.category_name,
    ).outerjoin(
        Inventory, Product.product_id == Inventory.product_id
    ).outerjoin(
        Category, Product.category_id == Category.category_id
    )

    if search:
        search_key = f'%{search}%'
        query = query.filter(
            db.or_(
                Product.product_name.like(search_key),
                Product.product_code.like(search_key),
                Product.barcode.like(search_key),
                Category.category_name.like(search_key),
            )
        )

    if category_id:
        query = query.filter(Product.category_id == category_id)

    if stock_state == 'out':
        query = query.filter(func.coalesce(Inventory.quantity, 0) <= 0)
    elif stock_state == 'low':
        query = query.filter(
            func.coalesce(Inventory.quantity, 0) > 0,
            func.coalesce(Inventory.quantity, 0) <= Product.min_stock,
        )
    elif stock_state == 'normal':
        query = query.filter(func.coalesce(Inventory.quantity, 0) > Product.min_stock)

    query = query.order_by(Product.product_id.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    items = []
    for product, quantity, category_name in pagination.items:
        quantity = int(quantity or 0)
        if quantity <= 0:
            stock_status = 'out'
        elif quantity <= product.min_stock:
            stock_status = 'low'
        else:
            stock_status = 'normal'

        items.append({
            'product_id': product.product_id,
            'product_code': product.product_code,
            'barcode': product.barcode or '',
            'product_name': product.product_name,
            'category_name': category_name or '-',
            'selling_price': float(product.selling_price),
            'purchase_price': float(product.purchase_price),
            'quantity': quantity,
            'min_stock': product.min_stock,
            'stock_status': stock_status,
            'stock_gap': max(product.min_stock - quantity, 0),
            'status': product.status,
            'updated_at': product.updated_at.strftime('%Y-%m-%d %H:%M:%S') if product.updated_at else '-',
        })

    return {
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page,
        'items': items,
    }


def get_inventory_logs(page=1, per_page=20, search='', change_type=None, start_date=None, end_date=None):
    """获取库存流水。"""
    query = db.session.query(
        InventoryLog,
        Product.product_name,
        Product.product_code,
        Product.barcode,
        User.username.label('operator_username'),
        User.real_name.label('operator_real_name'),
    ).join(
        Product, InventoryLog.product_id == Product.product_id
    ).outerjoin(
        User, InventoryLog.operator_id == User.user_id
    )

    if search:
        search_key = f'%{search}%'
        query = query.filter(
            db.or_(
                Product.product_name.like(search_key),
                Product.product_code.like(search_key),
                Product.barcode.like(search_key),
                InventoryLog.reason.like(search_key),
                User.username.like(search_key),
                User.real_name.like(search_key),
            )
        )

    if change_type:
        query = query.filter(InventoryLog.change_type == change_type)

    if start_date:
        query = query.filter(db.func.date(InventoryLog.created_at) >= start_date)

    if end_date:
        query = query.filter(db.func.date(InventoryLog.created_at) <= end_date)

    query = query.order_by(InventoryLog.created_at.desc(), InventoryLog.log_id.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    change_labels = {
        'in': '入库',
        'out': '出库',
        'adjust': '盘点调整',
        'sale': '销售出库',
        'return': '退货入库',
        'create': '商品新增',
        'import': '批量导入',
    }

    items = []
    for log, product_name, product_code, barcode, operator_username, operator_real_name in pagination.items:
        operator_name = operator_real_name or operator_username or '-'
        items.append({
            'log_id': log.log_id,
            'product_id': log.product_id,
            'product_name': product_name,
            'product_code': product_code,
            'barcode': barcode or '',
            'change_type': log.change_type,
            'change_type_label': change_labels.get(log.change_type, log.change_type),
            'quantity_change': int(log.quantity_change),
            'quantity_before': int(log.quantity_before),
            'quantity_after': int(log.quantity_after),
            'reason': log.reason or '-',
            'operator_name': operator_name,
            'created_at': log.created_at.strftime('%Y-%m-%d %H:%M:%S') if log.created_at else '-',
        })

    return {
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page,
        'items': items,
    }


def get_inventory_alerts(limit=20):
    """获取库存预警列表。"""
    results = db.session.query(
        Product,
        Inventory.quantity,
        Category.category_name,
    ).join(
        Inventory, Product.product_id == Inventory.product_id
    ).outerjoin(
        Category, Product.category_id == Category.category_id
    ).filter(
        Product.status == 1,
        Inventory.quantity <= Product.min_stock,
    ).order_by(
        Inventory.quantity.asc(),
        Product.product_id.desc(),
    ).limit(limit).all()

    alerts = []
    for product, quantity, category_name in results:
        quantity = int(quantity or 0)
        alerts.append({
            'product_id': product.product_id,
            'product_code': product.product_code,
            'product_name': product.product_name,
            'category_name': category_name or '-',
            'quantity': quantity,
            'min_stock': product.min_stock,
            'stock_gap': max(product.min_stock - quantity, 0),
            'alert_level': 'danger' if quantity <= 0 else 'warning',
            'updated_at': product.updated_at.strftime('%Y-%m-%d %H:%M:%S') if product.updated_at else '-',
        })

    return alerts

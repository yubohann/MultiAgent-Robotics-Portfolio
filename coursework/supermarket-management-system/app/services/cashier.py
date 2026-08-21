from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
import random

from sqlalchemy import func

from app import db
from app.models import Inventory, Product, Sale, SaleItem
from app.services.inventory import set_inventory_quantity

VALID_PAYMENT_METHODS = {'cash', 'wechat', 'alipay', 'card'}


def search_cashier_products(keyword='', limit=20):
    """收银台商品检索。"""
    keyword = (keyword or '').strip()
    limit = max(1, min(int(limit or 20), 100))

    query = db.session.query(
        Product.product_id,
        Product.product_code,
        Product.barcode,
        Product.product_name,
        Product.unit,
        Product.selling_price,
        func.coalesce(Inventory.quantity, 0).label('quantity'),
    ).outerjoin(
        Inventory, Product.product_id == Inventory.product_id
    ).filter(
        Product.status == 1,
        func.coalesce(Inventory.quantity, 0) > 0,
    )

    if keyword:
        like_key = f'%{keyword}%'
        query = query.filter(
            db.or_(
                Product.product_name.like(like_key),
                Product.product_code.like(like_key),
                Product.barcode.like(like_key),
            )
        )

    rows = query.order_by(Product.product_id.desc()).limit(limit).all()

    products = []
    for row in rows:
        products.append({
            'product_id': row.product_id,
            'product_code': row.product_code,
            'barcode': row.barcode or '',
            'product_name': row.product_name,
            'unit': row.unit or '件',
            'selling_price': float(row.selling_price),
            'quantity': int(row.quantity or 0),
        })

    return products


def checkout_cashier_order(cashier_id, items, payment_method='cash', discount_amount=0):
    """收银结算并生成订单。"""
    if payment_method not in VALID_PAYMENT_METHODS:
        return False, '支付方式不支持', None

    normalized_items = _normalize_cart_items(items)
    if not normalized_items:
        return False, '购物车为空，无法结算', None

    product_ids = [item['product_id'] for item in normalized_items]
    rows = db.session.query(
        Product.product_id,
        Product.product_code,
        Product.product_name,
        Product.selling_price,
        Product.purchase_price,
        Product.unit,
        Product.status,
        func.coalesce(Inventory.quantity, 0).label('quantity'),
    ).outerjoin(
        Inventory, Product.product_id == Inventory.product_id
    ).filter(
        Product.product_id.in_(product_ids)
    ).all()

    product_map = {row.product_id: row for row in rows}
    cart_details = []

    for item in normalized_items:
        row = product_map.get(item['product_id'])
        if not row:
            return False, f"商品不存在：ID {item['product_id']}", None
        if row.status != 1:
            return False, f"商品已下架：{row.product_name}", None

        stock = int(row.quantity or 0)
        if stock < item['quantity']:
            return False, f"库存不足：{row.product_name}（当前 {stock}）", None

        unit_price = Decimal(str(row.selling_price or 0))
        purchase_price = Decimal(str(row.purchase_price or 0))
        quantity = int(item['quantity'])
        subtotal = (unit_price * quantity).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

        cart_details.append({
            'product_id': row.product_id,
            'product_code': row.product_code,
            'product_name': row.product_name,
            'unit': row.unit or '件',
            'unit_price': unit_price,
            'purchase_price': purchase_price,
            'quantity': quantity,
            'subtotal': subtotal,
            'stock_before': stock,
        })

    total_amount = sum((item['subtotal'] for item in cart_details), Decimal('0.00'))
    discount = Decimal(str(discount_amount or 0)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    if discount < 0:
        return False, '优惠金额不能小于 0', None
    if discount > total_amount:
        return False, '优惠金额不能大于应收金额', None

    actual_amount = (total_amount - discount).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    try:
        order_no = _generate_order_no()
        sale = Sale(
            order_no=order_no,
            cashier_id=cashier_id,
            total_amount=total_amount,
            discount_amount=discount,
            actual_amount=actual_amount,
            payment_method=payment_method,
            status='completed',
        )
        db.session.add(sale)
        db.session.flush()

        for item in cart_details:
            db.session.add(SaleItem(
                sale_id=sale.sale_id,
                product_id=item['product_id'],
                quantity=item['quantity'],
                unit_price=item['unit_price'],
                purchase_price=item['purchase_price'],
                subtotal=item['subtotal'],
            ))

            quantity_after = item['stock_before'] - item['quantity']
            set_inventory_quantity(
                product_id=item['product_id'],
                quantity_after=quantity_after,
                change_type='sale',
                reason=f"销售订单 {order_no}",
                operator_id=cashier_id,
            )

        db.session.commit()

        order = {
            'sale_id': sale.sale_id,
            'order_no': order_no,
            'total_amount': float(total_amount),
            'discount_amount': float(discount),
            'actual_amount': float(actual_amount),
            'payment_method': payment_method,
            'created_at': sale.created_at.strftime('%Y-%m-%d %H:%M:%S') if sale.created_at else '',
            'items': [
                {
                    'product_id': item['product_id'],
                    'product_code': item['product_code'],
                    'product_name': item['product_name'],
                    'unit': item['unit'],
                    'quantity': item['quantity'],
                    'unit_price': float(item['unit_price']),
                    'subtotal': float(item['subtotal']),
                }
                for item in cart_details
            ],
        }
        return True, '结算成功', order
    except Exception as e:
        db.session.rollback()
        return False, f'结算失败：{str(e)}', None


def _normalize_cart_items(items):
    if not isinstance(items, list):
        return []

    merged = {}
    for row in items:
        if not isinstance(row, dict):
            continue

        try:
            product_id = int(row.get('product_id'))
            quantity = int(row.get('quantity'))
        except (TypeError, ValueError):
            continue

        if product_id <= 0 or quantity <= 0:
            continue

        merged[product_id] = merged.get(product_id, 0) + quantity

    normalized = []
    for product_id, quantity in merged.items():
        normalized.append({'product_id': product_id, 'quantity': quantity})
    return normalized


def _generate_order_no():
    now = datetime.now()
    time_part = now.strftime('%Y%m%d%H%M%S')
    random_part = random.randint(1000, 9999)
    return f'SO{time_part}{random_part}'

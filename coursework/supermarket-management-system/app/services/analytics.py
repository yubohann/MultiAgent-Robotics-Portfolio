from app import db
from app.models import Category, Inventory, Product, Sale, SaleItem
from sqlalchemy import func


def get_dashboard_overview():
    """获取首页概览统计"""
    today_start = db.func.date(db.func.datetime('now', 'localtime'))

    total_products = Product.query.count()
    total_inventory = db.session.query(func.coalesce(func.sum(Inventory.quantity), 0)).scalar() or 0

    today_sales = db.session.query(func.coalesce(func.sum(Sale.actual_amount), 0)).filter(
        Sale.status == 'completed',
        db.func.date(Sale.created_at) == today_start,
    ).scalar() or 0

    low_stock_count = db.session.query(func.count(Product.product_id)).join(
        Inventory, Product.product_id == Inventory.product_id
    ).filter(
        Product.status == 1,
        Inventory.quantity <= Product.min_stock,
    ).scalar() or 0

    return {
        'total_products': int(total_products),
        'total_inventory': int(total_inventory),
        'today_sales': round(float(today_sales), 2),
        'low_stock_count': int(low_stock_count),
    }


def get_sales_overview(period='month'):
    """获取销售概览统计数据"""
    now = db.func.datetime('now', 'localtime')
    if period == 'today':
        start_date = db.func.date(now)
        prev_start = db.func.date(db.func.datetime(now, '-1 day'))
        prev_end = prev_start
    elif period == 'week':
        start_date = db.func.date(db.func.datetime(now, '-7 days'))
        prev_start = db.func.date(db.func.datetime(now, '-14 days'))
        prev_end = db.func.date(db.func.datetime(now, '-8 days'))
    else:
        start_date = db.func.strftime('%Y-%m-01', now)
        prev_start = db.func.strftime('%Y-%m-01', db.func.datetime(now, '-1 month'))
        prev_end = db.func.strftime('%Y-%m-%d', db.func.datetime(db.func.strftime('%Y-%m-01', now), '-1 day'))

    current_stats = db.session.query(
        func.coalesce(func.sum(Sale.actual_amount), 0).label('total_sales'),
        func.count(Sale.sale_id).label('order_count'),
        func.coalesce(func.avg(Sale.actual_amount), 0).label('avg_order_value'),
    ).filter(
        Sale.status == 'completed',
        db.func.date(Sale.created_at) >= start_date,
    ).first()

    gross_profit_result = db.session.query(
        func.coalesce(func.sum((SaleItem.unit_price - SaleItem.purchase_price) * SaleItem.quantity), 0)
    ).join(Sale).filter(
        Sale.status == 'completed',
        db.func.date(Sale.created_at) >= start_date,
    ).first()

    total_sales = float(current_stats.total_sales)
    order_count = current_stats.order_count
    avg_order_value = float(current_stats.avg_order_value) if order_count > 0 else 0
    gross_profit = float(gross_profit_result[0])
    profit_rate = (gross_profit / total_sales * 100) if total_sales > 0 else 0

    prev_stats = db.session.query(
        func.coalesce(func.sum(Sale.actual_amount), 0).label('total_sales')
    ).filter(
        Sale.status == 'completed',
        db.func.date(Sale.created_at) >= prev_start,
        db.func.date(Sale.created_at) <= prev_end,
    ).first()

    prev_total_sales = float(prev_stats.total_sales)
    growth_rate = ((total_sales - prev_total_sales) / prev_total_sales * 100) if prev_total_sales > 0 else 0

    return {
        'total_sales': round(total_sales, 2),
        'order_count': order_count,
        'avg_order_value': round(avg_order_value, 2),
        'gross_profit': round(gross_profit, 2),
        'profit_rate': round(profit_rate, 2),
        'growth_rate': round(growth_rate, 2),
    }


def get_sales_trend(days=30):
    """获取每日销售趋势数据"""
    now = db.func.datetime('now', 'localtime')
    start_date = db.func.date(db.func.datetime(now, f'-{days} days'))

    results = db.session.query(
        db.func.date(Sale.created_at).label('sale_date'),
        func.coalesce(func.sum(Sale.actual_amount), 0).label('daily_sales'),
        func.count(Sale.sale_id).label('order_count'),
    ).filter(
        Sale.status == 'completed',
        db.func.date(Sale.created_at) >= start_date,
    ).group_by(
        db.func.date(Sale.created_at)
    ).order_by(
        db.func.date(Sale.created_at)
    ).all()

    trend_data = []
    for row in results:
        trend_data.append({
            'date': str(row.sale_date),
            'sales': round(float(row.daily_sales), 2),
            'orders': row.order_count,
        })

    return trend_data


def get_top_products(limit=10, period='month', sort_by='quantity'):
    """获取热销/滞销商品排行"""
    now = db.func.datetime('now', 'localtime')
    if period == 'today':
        start_date = db.func.date(now)
    elif period == 'week':
        start_date = db.func.date(db.func.datetime(now, '-7 days'))
    else:
        start_date = db.func.strftime('%Y-%m-01', now)

    total_sales_result = db.session.query(
        func.coalesce(func.sum(SaleItem.subtotal), 0)
    ).join(Sale).filter(
        Sale.status == 'completed',
        db.func.date(Sale.created_at) >= start_date,
    ).first()
    total_sales = float(total_sales_result[0]) if total_sales_result[0] else 1

    query = db.session.query(
        Product.product_name,
        func.coalesce(func.sum(SaleItem.quantity), 0).label('total_qty'),
        func.coalesce(func.sum(SaleItem.subtotal), 0).label('total_sales')
    ).join(SaleItem).join(Sale).filter(
        Sale.status == 'completed',
        db.func.date(Sale.created_at) >= start_date,
    ).group_by(
        Product.product_id, Product.product_name
    )

    if sort_by == 'quantity':
        query = query.order_by(func.sum(SaleItem.quantity).desc())
    elif sort_by == 'sales':
        query = query.order_by(func.sum(SaleItem.subtotal).desc())
    elif sort_by == 'quantity_asc':
        query = query.order_by(func.sum(SaleItem.quantity).asc())
    elif sort_by == 'sales_asc':
        query = query.order_by(func.sum(SaleItem.subtotal).asc())

    results = query.limit(limit).all()

    products_data = []
    for idx, row in enumerate(results, 1):
        qty = row.total_qty
        sales = float(row.total_sales)
        percentage = round((sales / total_sales * 100), 2) if total_sales > 0 else 0

        products_data.append({
            'rank': idx,
            'product_name': row.product_name,
            'quantity_sold': qty,
            'total_sales': round(sales, 2),
            'percentage': f'{percentage}%'
        })

    return products_data


def get_category_distribution(period='month'):
    """获取商品分类销售占比"""
    now = db.func.datetime('now', 'localtime')
    if period == 'today':
        start_date = db.func.date(now)
    elif period == 'week':
        start_date = db.func.date(db.func.datetime(now, '-7 days'))
    else:
        start_date = db.func.strftime('%Y-%m-01', now)

    results = db.session.query(
        Category.category_name,
        func.coalesce(func.sum(SaleItem.subtotal), 0).label('category_sales')
    ).select_from(SaleItem).join(Sale).join(Product).join(Category).filter(
        Sale.status == 'completed',
        db.func.date(Sale.created_at) >= start_date,
    ).group_by(
        Category.category_id, Category.category_name
    ).order_by(
        func.sum(SaleItem.subtotal).desc()
    ).all()

    total_sales = sum([float(row.category_sales) for row in results])

    distribution = []
    for row in results:
        sales = float(row.category_sales)
        percentage = round((sales / total_sales * 100), 2) if total_sales > 0 else 0

        distribution.append({
            'category_name': row.category_name if row.category_name else '未分类',
            'sales': round(sales, 2),
            'percentage': percentage,
        })

    return distribution

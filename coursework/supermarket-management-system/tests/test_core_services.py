from datetime import date
from io import BytesIO

from openpyxl import Workbook

from app import db
from app.models import Announcement, Inventory, InventoryLog, Product
from app.services.announcements import (
    create_announcement,
    get_unread_announcement_count,
    mark_announcement_read,
)
from app.services.cashier import checkout_cashier_order
from app.services.finance import get_reconciliation, save_reconciliation
from app.services.inventory import set_inventory_quantity
from app.services.products import create_product, import_products_from_excel
from app.services.sales import get_sales_order_detail


def test_product_inventory_checkout_finance_and_announcement_flow(app, users, default_category):
    with app.app_context():
        success, message = create_product({
            'product_code': 'P9001',
            'barcode': '6900009001',
            'product_name': '测试饮料',
            'category_id': default_category,
            'unit': '瓶',
            'purchase_price': 2.0,
            'selling_price': 5.0,
            'quantity': 10,
            'min_stock': 3,
        })
        assert success is True
        assert message == '商品创建成功'

        success, message = create_product({
            'product_code': 'P9001',
            'product_name': '重复编码',
            'selling_price': 6.0,
        })
        assert success is False
        assert message == '商品编码已存在'

        product = Product.query.filter_by(product_code='P9001').first()
        inventory = db.session.get(Inventory, product.product_id)
        assert inventory.quantity == 10

        set_inventory_quantity(product.product_id, 8, change_type='adjust', reason='测试盘点', operator_id=users['admin_id'])
        assert db.session.get(Inventory, product.product_id).quantity == 8
        assert InventoryLog.query.filter_by(product_id=product.product_id, quantity_after=8).first() is not None

        success, message, order = checkout_cashier_order(
            users['cashier_id'],
            [{'product_id': product.product_id, 'quantity': 20}],
            payment_method='cash',
        )
        assert success is False
        assert '库存不足' in message
        assert order is None

        success, message, order = checkout_cashier_order(
            users['cashier_id'],
            [{'product_id': product.product_id, 'quantity': 2}],
            payment_method='cash',
            discount_amount=1,
        )
        assert success is True
        assert message == '结算成功'
        assert order['actual_amount'] == 9.0
        assert db.session.get(Inventory, product.product_id).quantity == 6

        detail = get_sales_order_detail(order['sale_id'])
        assert detail['order_no'] == order['order_no']
        assert detail['item_count'] == 1

        today = date.today().strftime('%Y-%m-%d')
        success, message = save_reconciliation(today, 'cash', 9.0, '测试对账', users['admin_id'])
        assert success is True
        assert message == '对账保存成功'
        reconciliation = get_reconciliation(today)
        assert reconciliation['summary']['expected_total'] == 9.0
        assert reconciliation['summary']['difference_total'] == 0.0

        success, message = create_announcement(
            title='测试公告',
            content='今晚盘点',
            created_by=users['admin_id'],
            target_role='cashier',
        )
        assert success is True
        announcement = Announcement.query.filter_by(title='测试公告').first()
        assert get_unread_announcement_count(users['cashier_id'], 'cashier') == 1

        success, message = mark_announcement_read(users['cashier_id'], 'cashier', announcement.announcement_id)
        assert success is True
        assert message == '标记成功'
        assert get_unread_announcement_count(users['cashier_id'], 'cashier') == 0


def test_import_products_from_excel_creates_inventory_log(app):
    with app.app_context():
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.append(['product_code', 'product_name', 'selling_price', 'quantity', 'min_stock'])
        worksheet.append(['P9101', 'Excel导入商品', 12.5, 7, 2])

        buffer = BytesIO()
        workbook.save(buffer)
        workbook.close()

        success_count, error_count, errors = import_products_from_excel(buffer.getvalue())

        assert success_count == 1
        assert error_count == 0
        assert errors == []

        product = Product.query.filter_by(product_code='P9101').first()
        assert product is not None
        inventory = db.session.get(Inventory, product.product_id)
        assert inventory.quantity == 7
        log = InventoryLog.query.filter_by(product_id=product.product_id).first()
        assert log.change_type == 'import'
        assert log.quantity_after == 7

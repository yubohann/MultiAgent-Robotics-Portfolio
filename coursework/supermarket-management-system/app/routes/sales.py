from flask import jsonify, render_template, request

from app.models import User
from app.routes.common import admin_required
from app.services.sales import get_sales_order_detail, get_sales_orders


def register_routes(app):
    @app.route('/sales')
    @admin_required
    def sales():
        cashiers = User.query.filter_by(role='cashier').order_by(User.user_id).all()
        return render_template('sales.html', active_page='sales', cashiers=cashiers)

    @app.route('/api/sales/orders', methods=['GET'])
    @admin_required
    def api_sales_orders():
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        search = request.args.get('search', '').strip()
        cashier_id = request.args.get('cashier_id', type=int)
        status = request.args.get('status', '').strip() or None
        payment_method = request.args.get('payment_method', '').strip() or None
        start_date = request.args.get('start_date', '').strip() or None
        end_date = request.args.get('end_date', '').strip() or None

        data = get_sales_orders(
            page=page,
            per_page=per_page,
            search=search,
            cashier_id=cashier_id,
            status=status,
            payment_method=payment_method,
            start_date=start_date,
            end_date=end_date,
        )
        return jsonify(data)

    @app.route('/api/sales/orders/<int:sale_id>', methods=['GET'])
    @admin_required
    def api_sales_order_detail(sale_id):
        data = get_sales_order_detail(sale_id)
        if not data:
            return jsonify({'success': False, 'message': '订单不存在'}), 404
        return jsonify({'success': True, 'order': data})

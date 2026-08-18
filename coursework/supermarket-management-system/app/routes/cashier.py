from flask import jsonify, render_template, request, session

from app.routes.common import cashier_required
from app.services.cashier import checkout_cashier_order, search_cashier_products


def register_routes(app):
    @app.route('/cashier')
    @cashier_required
    def cashier_page():
        return render_template('cashier.html', active_page='cashier')

    @app.route('/api/cashier/products', methods=['GET'])
    @cashier_required
    def api_cashier_products():
        keyword = request.args.get('keyword', '').strip()
        limit = request.args.get('limit', 20, type=int)
        products = search_cashier_products(keyword=keyword, limit=limit)
        return jsonify({'success': True, 'products': products})

    @app.route('/api/cashier/checkout', methods=['POST'])
    @cashier_required
    def api_cashier_checkout():
        payload = request.get_json(silent=True) or {}
        items = payload.get('items', [])
        payment_method = str(payload.get('payment_method', 'cash')).strip() or 'cash'
        discount_amount = payload.get('discount_amount', 0)

        success, message, order = checkout_cashier_order(
            cashier_id=session.get('user_id'),
            items=items,
            payment_method=payment_method,
            discount_amount=discount_amount,
        )

        if not success:
            return jsonify({'success': False, 'message': message}), 400

        return jsonify({'success': True, 'message': message, 'order': order})

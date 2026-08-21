from flask import jsonify, render_template, request

from app.routes.common import admin_required
from app.services.inventory import get_inventory_alerts, get_inventory_list, get_inventory_logs, get_inventory_summary


def register_routes(app):
    @app.route('/inventory')
    @admin_required
    def inventory():
        return render_template('inventory.html', active_page='inventory')

    @app.route('/api/inventory/summary', methods=['GET'])
    @admin_required
    def api_inventory_summary():
        return jsonify({'success': True, 'summary': get_inventory_summary()})

    @app.route('/api/inventory/list', methods=['GET'])
    @admin_required
    def api_inventory_list():
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        search = request.args.get('search', '').strip()
        category_id = request.args.get('category_id', type=int)
        stock_state = request.args.get('stock_state', '').strip() or None

        data = get_inventory_list(
            page=page,
            per_page=per_page,
            search=search,
            category_id=category_id,
            stock_state=stock_state,
        )
        return jsonify({'success': True, **data})

    @app.route('/api/inventory/logs', methods=['GET'])
    @admin_required
    def api_inventory_logs():
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        search = request.args.get('search', '').strip()
        change_type = request.args.get('change_type', '').strip() or None
        start_date = request.args.get('start_date', '').strip() or None
        end_date = request.args.get('end_date', '').strip() or None

        data = get_inventory_logs(
            page=page,
            per_page=per_page,
            search=search,
            change_type=change_type,
            start_date=start_date,
            end_date=end_date,
        )
        return jsonify({'success': True, **data})

    @app.route('/api/inventory/alerts', methods=['GET'])
    @admin_required
    def api_inventory_alerts():
        limit = request.args.get('limit', 20, type=int)
        alerts = get_inventory_alerts(limit=limit)
        return jsonify({'success': True, 'alerts': alerts, 'total': len(alerts)})

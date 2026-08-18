from flask import jsonify, render_template, request

from app.routes.common import admin_required
from app.services.analytics import get_category_distribution, get_sales_overview, get_sales_trend, get_top_products


def register_routes(app):
    @app.route('/analytics')
    @admin_required
    def analytics():
        return render_template('analytics.html', active_page='analytics')

    @app.route('/api/analytics/overview')
    @admin_required
    def api_analytics_overview():
        period = request.args.get('period', 'month')
        data = get_sales_overview(period)
        return jsonify(data)

    @app.route('/api/analytics/trend')
    @admin_required
    def api_analytics_trend():
        days = request.args.get('days', 30, type=int)
        data = get_sales_trend(days)
        return jsonify({'trend': data})

    @app.route('/api/analytics/top-products')
    @admin_required
    def api_analytics_top_products():
        limit = request.args.get('limit', 10, type=int)
        period = request.args.get('period', 'month')
        sort_by = request.args.get('sort_by', 'quantity')
        data = get_top_products(limit, period, sort_by)
        return jsonify({'products': data})

    @app.route('/api/analytics/category-distribution')
    @admin_required
    def api_analytics_category_distribution():
        period = request.args.get('period', 'month')
        data = get_category_distribution(period)
        return jsonify({'categories': data})

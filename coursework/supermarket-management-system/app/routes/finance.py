from flask import jsonify, render_template, request, session

from app.routes.common import admin_required
from app.services.finance import (
    close_finance_period,
    create_finance_transaction,
    create_payable,
    get_finance_overview,
    get_finance_transactions,
    get_payables,
    get_reconciliation,
    get_recent_closings,
    record_payable_payment,
    save_reconciliation,
)


def register_routes(app):
    @app.route('/finance')
    @admin_required
    def finance_page():
        return render_template('finance.html', active_page='finance')

    @app.route('/api/finance/overview', methods=['GET'])
    @admin_required
    def api_finance_overview():
        period = request.args.get('period', 'month')
        data = get_finance_overview(period=period)
        return jsonify({'success': True, 'overview': data})

    @app.route('/api/finance/transactions', methods=['GET'])
    @admin_required
    def api_finance_transactions():
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        transaction_type = request.args.get('transaction_type', '').strip() or None
        category = request.args.get('category', '').strip()
        search = request.args.get('search', '').strip()
        start_date = request.args.get('start_date', '').strip() or None
        end_date = request.args.get('end_date', '').strip() or None

        data = get_finance_transactions(
            page=page,
            per_page=per_page,
            transaction_type=transaction_type,
            category=category,
            search=search,
            start_date=start_date,
            end_date=end_date,
        )
        return jsonify({'success': True, **data})

    @app.route('/api/finance/transactions', methods=['POST'])
    @admin_required
    def api_finance_create_transaction():
        payload = request.get_json(silent=True) or {}
        success, message = create_finance_transaction(
            transaction_type=payload.get('transaction_type'),
            category=payload.get('category'),
            amount=payload.get('amount'),
            payment_method=payload.get('payment_method'),
            occurred_at=payload.get('occurred_at'),
            description=payload.get('description'),
            related_order_no=payload.get('related_order_no'),
            operator_id=session.get('user_id'),
        )
        return jsonify({'success': success, 'message': message}), 200 if success else 400

    @app.route('/api/finance/reconciliation', methods=['GET'])
    @admin_required
    def api_finance_reconciliation():
        date_text = request.args.get('date', '').strip() or None
        data = get_reconciliation(date_text)
        return jsonify({'success': True, **data})

    @app.route('/api/finance/reconciliation', methods=['POST'])
    @admin_required
    def api_finance_save_reconciliation():
        payload = request.get_json(silent=True) or {}
        success, message = save_reconciliation(
            date_text=payload.get('date'),
            payment_method=payload.get('payment_method'),
            actual_amount=payload.get('actual_amount'),
            note=payload.get('note', ''),
            operator_id=session.get('user_id'),
        )
        return jsonify({'success': success, 'message': message}), 200 if success else 400

    @app.route('/api/finance/payables', methods=['GET'])
    @admin_required
    def api_finance_payables():
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        status = request.args.get('status', '').strip() or None
        search = request.args.get('search', '').strip()
        due_start = request.args.get('due_start', '').strip() or None
        due_end = request.args.get('due_end', '').strip() or None

        data = get_payables(
            page=page,
            per_page=per_page,
            status=status,
            search=search,
            due_start=due_start,
            due_end=due_end,
        )
        return jsonify({'success': True, **data})

    @app.route('/api/finance/payables', methods=['POST'])
    @admin_required
    def api_finance_create_payable():
        payload = request.get_json(silent=True) or {}
        success, message = create_payable(
            supplier_name=payload.get('supplier_name'),
            total_amount=payload.get('total_amount'),
            due_date=payload.get('due_date'),
            bill_no=payload.get('bill_no'),
            note=payload.get('note'),
            created_by=session.get('user_id'),
        )
        return jsonify({'success': success, 'message': message}), 200 if success else 400

    @app.route('/api/finance/payables/<int:payable_id>/payment', methods=['POST'])
    @admin_required
    def api_finance_payable_payment(payable_id):
        payload = request.get_json(silent=True) or {}
        success, message = record_payable_payment(
            payable_id=payable_id,
            amount=payload.get('amount'),
            payment_method=payload.get('payment_method', 'bank_transfer'),
            paid_at=payload.get('paid_at'),
            remark=payload.get('remark'),
            operator_id=session.get('user_id'),
        )
        return jsonify({'success': success, 'message': message}), 200 if success else 400

    @app.route('/api/finance/closings', methods=['GET'])
    @admin_required
    def api_finance_closings():
        limit = request.args.get('limit', 6, type=int)
        rows = get_recent_closings(limit=limit)
        return jsonify({'success': True, 'items': rows})

    @app.route('/api/finance/close-month', methods=['POST'])
    @admin_required
    def api_finance_close_month():
        payload = request.get_json(silent=True) or {}
        success, message, snapshot = close_finance_period(
            period_month=payload.get('period_month'),
            note=payload.get('note', ''),
            operator_id=session.get('user_id'),
        )

        response = {'success': success, 'message': message}
        if snapshot:
            response['snapshot'] = snapshot
        return jsonify(response), 200 if success else 400

from flask import flash, jsonify, redirect, render_template, request, session, url_for

from app.models import User
from app.routes.common import admin_required
from app.services.analytics import get_dashboard_overview
from app.services.auth import (
    get_admin_signup_requests,
    login_user,
    register_user,
    review_admin_signup_request,
)


def register_routes(app):
    @app.route('/')
    @admin_required
    def index():
        stats = get_dashboard_overview()
        return render_template('index.html', active_page='index', stats=stats)

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if 'user_id' in session:
            if session.get('role') == 'cashier':
                return redirect(url_for('cashier_page'))
            return redirect(url_for('index'))

        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')

            success, message, user = login_user(username, password)

            if success:
                session['user_id'] = user.user_id
                session['username'] = user.username
                session['role'] = user.role
                flash(message, 'success')

                if user.role == 'cashier':
                    return redirect(url_for('cashier_page'))
                return redirect(url_for('index'))

            flash(message, 'error')
            return redirect(url_for('login'))

        return render_template('login.html')

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if 'user_id' in session:
            if session.get('role') == 'cashier':
                return redirect(url_for('cashier_page'))
            return redirect(url_for('index'))

        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            role = request.form.get('role', 'cashier')

            success, message = register_user(username, password, confirm_password, role)

            if success:
                flash(message, 'success')
                return redirect(url_for('login'))

            flash(message, 'error')
            return redirect(url_for('register'))

        return render_template('register.html')

    @app.route('/admin/register-requests')
    @admin_required
    def admin_register_requests_page():
        status = request.args.get('status', 'pending')
        signup_requests = get_admin_signup_requests(status)
        return render_template(
            'admin_register_requests.html',
            active_page='admin_register_requests',
            signup_requests=signup_requests,
            current_status=status,
        )

    @app.route('/admin/register-requests/<int:request_id>/approve', methods=['POST'])
    @admin_required
    def approve_admin_register_request(request_id):
        reviewer_id = session.get('user_id')
        success, message = review_admin_signup_request(request_id, reviewer_id, 'approve')

        flash(message, 'success' if success else 'error')
        return redirect(url_for('admin_register_requests_page', status='pending'))

    @app.route('/admin/register-requests/<int:request_id>/reject', methods=['POST'])
    @admin_required
    def reject_admin_register_request(request_id):
        reviewer_id = session.get('user_id')
        reject_reason = request.form.get('reject_reason')
        success, message = review_admin_signup_request(
            request_id,
            reviewer_id,
            'reject',
            reject_reason=reject_reason,
        )

        flash(message, 'success' if success else 'error')
        return redirect(url_for('admin_register_requests_page', status='pending'))

    @app.route('/logout')
    def logout():
        session.clear()
        flash('已退出登录', 'success')
        return redirect(url_for('login'))

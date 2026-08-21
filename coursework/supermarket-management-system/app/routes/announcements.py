from flask import flash, jsonify, redirect, render_template, request, session, url_for

from app.routes.common import admin_required, login_required
from app.services.announcements import (
    create_announcement,
    get_admin_announcements,
    get_announcements_for_user,
    get_unread_announcement_count,
    mark_all_announcements_read,
    mark_announcement_read,
    set_announcement_publish_status,
)


def register_routes(app):
    @app.route('/announcements')
    @admin_required
    def announcements_page():
        announcements = get_admin_announcements()
        return render_template(
            'announcements.html',
            active_page='announcements',
            announcements=announcements,
        )

    @app.route('/announcements/create', methods=['POST'])
    @admin_required
    def create_announcement_action():
        title = request.form.get('title')
        content = request.form.get('content')
        level = request.form.get('level', 'normal')
        target_role = request.form.get('target_role', 'all')
        is_published = request.form.get('is_published', '1')

        success, message = create_announcement(
            title=title,
            content=content,
            created_by=session.get('user_id'),
            level=level,
            target_role=target_role,
            is_published=is_published,
        )

        flash(message, 'success' if success else 'error')
        return redirect(url_for('announcements_page'))

    @app.route('/announcements/<int:announcement_id>/toggle', methods=['POST'])
    @admin_required
    def toggle_announcement_action(announcement_id):
        is_published = request.form.get('is_published', '1')
        success, message = set_announcement_publish_status(announcement_id, is_published)
        flash(message, 'success' if success else 'error')
        return redirect(url_for('announcements_page'))

    @app.route('/api/announcements')
    @login_required
    def get_announcements_api():
        limit_raw = request.args.get('limit', '8')
        try:
            limit = int(limit_raw)
        except (TypeError, ValueError):
            limit = 8

        user_id = session.get('user_id')
        role = session.get('role')

        data = get_announcements_for_user(user_id=user_id, role=role, limit=limit)
        return jsonify({'success': True, 'data': data})

    @app.route('/api/announcements/unread-count')
    @login_required
    def get_unread_count_api():
        user_id = session.get('user_id')
        role = session.get('role')
        unread_count = get_unread_announcement_count(user_id=user_id, role=role)
        return jsonify({'success': True, 'unread_count': unread_count})

    @app.route('/api/announcements/<int:announcement_id>/read', methods=['POST'])
    @login_required
    def mark_announcement_read_api(announcement_id):
        user_id = session.get('user_id')
        role = session.get('role')
        success, message = mark_announcement_read(
            user_id=user_id,
            role=role,
            announcement_id=announcement_id,
        )

        status_code = 200 if success else 400
        return jsonify({'success': success, 'message': message}), status_code

    @app.route('/api/announcements/read-all', methods=['POST'])
    @login_required
    def mark_all_announcements_read_api():
        user_id = session.get('user_id')
        role = session.get('role')
        success, message, changed_count = mark_all_announcements_read(user_id=user_id, role=role)

        status_code = 200 if success else 400
        return jsonify({
            'success': success,
            'message': message,
            'changed_count': changed_count,
        }), status_code

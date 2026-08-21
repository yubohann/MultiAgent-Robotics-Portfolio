from flask import jsonify, render_template, request

from app.routes.common import admin_required
from app.services.assistant import generate_assistant_reply


def register_routes(app):
    @app.route('/assistant')
    @admin_required
    def assistant():
        return render_template('assistant.html', active_page='assistant')

    @app.route('/api/assistant/chat', methods=['POST'])
    @admin_required
    def api_assistant_chat():
        payload = request.get_json(silent=True) or {}
        message = str(payload.get('message', '')).strip()

        result = generate_assistant_reply(message)
        status_code = 200 if result.get('success') else 400
        return jsonify(result), status_code

from flask import jsonify, render_template, request

from app.routes.common import admin_required
from app.services.products import (
    create_product,
    delete_product,
    get_categories,
    get_products,
    import_products_from_csv,
    import_products_from_excel,
    offline_product,
    online_product,
    update_product,
)


def register_routes(app):
    @app.route('/product')
    @admin_required
    def product():
        return render_template('product.html', active_page='product')

    @app.route('/api/products', methods=['GET'])
    @admin_required
    def api_products():
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        search = request.args.get('search', '')
        category_id = request.args.get('category_id', type=int)
        status = request.args.get('status', type=int)

        result = get_products(page, per_page, search, category_id, status)
        return jsonify(result)

    @app.route('/api/products', methods=['POST'])
    @admin_required
    def api_create_product():
        data = request.get_json()
        success, message = create_product(data)
        return jsonify({'success': success, 'message': message})

    @app.route('/api/products/<int:product_id>', methods=['PUT'])
    @admin_required
    def api_update_product(product_id):
        data = request.get_json()
        success, message = update_product(product_id, data)
        return jsonify({'success': success, 'message': message})

    @app.route('/api/products/<int:product_id>/offline', methods=['POST'])
    @admin_required
    def api_offline_product(product_id):
        success, message = offline_product(product_id)
        return jsonify({'success': success, 'message': message})

    @app.route('/api/products/<int:product_id>/online', methods=['POST'])
    @admin_required
    def api_online_product(product_id):
        success, message = online_product(product_id)
        return jsonify({'success': success, 'message': message})

    @app.route('/api/products/<int:product_id>', methods=['DELETE'])
    @admin_required
    def api_delete_product(product_id):
        success, message = delete_product(product_id)
        return jsonify({'success': success, 'message': message})

    @app.route('/api/products/import', methods=['POST'])
    @admin_required
    def api_import_products():
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': '未选择文件'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': '未选择文件'})

        filename = file.filename.lower()
        file_content = file.read()

        try:
            if filename.endswith('.csv'):
                success_count, error_count, errors = import_products_from_csv(file_content)
            elif filename.endswith(('.xlsx', '.xls')):
                success_count, error_count, errors = import_products_from_excel(file_content)
            else:
                return jsonify({'success': False, 'message': '不支持的文件格式，请上传 CSV 或 Excel 文件'})

            message = f'导入完成：成功 {success_count} 条'
            if error_count > 0:
                message += f'，失败 {error_count} 条'

            return jsonify({
                'success': True,
                'message': message,
                'success_count': success_count,
                'error_count': error_count,
                'errors': errors[:10],
            })
        except Exception as e:
            return jsonify({'success': False, 'message': f'导入失败：{str(e)}'})

    @app.route('/api/categories', methods=['GET'])
    @admin_required
    def api_categories():
        categories = get_categories()
        return jsonify({'categories': categories})

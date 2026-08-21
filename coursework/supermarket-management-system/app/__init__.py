import logging
import os
import time
from logging.handlers import RotatingFileHandler

from flask import Flask
from flask import g, has_request_context, request
from flask_sqlalchemy import SQLAlchemy
from config import Config

db = SQLAlchemy()


def _setup_logging(app):
    """配置应用日志，输出到 app.log。"""
    base_dir = app.config.get('BASE_DIR', app.root_path)
    log_file = app.config.get('LOG_FILE') or os.path.join(base_dir, 'log', 'app.log')
    if not os.path.isabs(log_file):
        log_file = os.path.join(base_dir, log_file)
    log_level_name = str(app.config.get('LOG_LEVEL', 'INFO')).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    max_bytes = int(app.config.get('LOG_MAX_BYTES', 5 * 1024 * 1024))
    backup_count = int(app.config.get('LOG_BACKUP_COUNT', 5))

    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    app.logger.setLevel(log_level)

    existing_handler = any(
        isinstance(handler, RotatingFileHandler) and os.path.abspath(getattr(handler, 'baseFilename', '')) == os.path.abspath(log_file)
        for handler in app.logger.handlers
    )
    if not existing_handler:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8',
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s %(message)s'
        ))
        app.logger.addHandler(file_handler)

    @app.before_request
    def _log_request_start():
        g.request_started_at = time.perf_counter()

    @app.after_request
    def _log_request_end(response):
        started_at = getattr(g, 'request_started_at', None)
        elapsed_ms = 0.0
        if started_at is not None:
            elapsed_ms = (time.perf_counter() - started_at) * 1000

        app.logger.info(
            'request method=%s path=%s status=%s elapsed_ms=%.2f ip=%s',
            request.method,
            request.path,
            response.status_code,
            elapsed_ms,
            request.remote_addr,
        )
        return response

    @app.teardown_request
    def _log_unhandled_exception(exc):
        if exc is not None and has_request_context():
            app.logger.exception(
                'unhandled_exception method=%s path=%s ip=%s',
                request.method,
                request.path,
                request.remote_addr,
            )

    app.logger.info('logging_initialized file=%s level=%s', log_file, log_level_name)


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)

    _setup_logging(app)

    db.init_app(app)

    with app.app_context():
        from app.models import (
            User,
            AdminSignupRequest,
            Announcement,
            AnnouncementRead,
            Category,
            Product,
            Inventory,
            InventoryLog,
            Sale,
            SaleItem,
            FinanceTransaction,
            CashReconciliation,
            SupplierPayable,
            PayablePayment,
            FinancePeriodClose,
            Member,
            Employee,
            Supplier,
            SystemSetting,
        )
        db.create_all()
        
        # 初始化默认数据
        if app.config.get('INIT_DEFAULT_DATA', True):
            _init_default_users()
            _init_default_categories()
            _init_second_phase_defaults()
            _init_seed_data_if_empty(app)

    # 注册路由
    from app.routes import register_routes
    register_routes(app)

    return app


def _init_default_users():
    """初始化默认用户"""
    from app.models import User
    
    # 检查是否已有管理员
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            real_name='系统管理员',
            role='admin',
            is_active=1
        )
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
    
    # 检查是否已有收银员
    if not User.query.filter_by(username='cashier01').first():
        cashier = User(
            username='cashier01',
            real_name='收银员01',
            role='cashier',
            is_active=1
        )
        cashier.set_password('123456')
        db.session.add(cashier)
        db.session.commit()


def _init_default_categories():
    """初始化默认分类"""
    from app.models import Category
    
    default_categories = [
        {'category_id': 1, 'category_name': '食品饮料', 'sort_order': 1},
        {'category_id': 2, 'category_name': '日用百货', 'sort_order': 2},
        {'category_id': 3, 'category_name': '生鲜果蔬', 'sort_order': 3},
        {'category_id': 4, 'category_name': '零食糖果', 'sort_order': 4},
        {'category_id': 5, 'category_name': '烟酒专区', 'sort_order': 5},
    ]
    
    for cat_data in default_categories:
        if not db.session.get(Category, cat_data['category_id']):
            category = Category(**cat_data)
            db.session.add(category)
    
    db.session.commit()


def _init_second_phase_defaults():
    """初始化会员、员工、供应商和系统参数演示数据。"""
    from app.models import Employee, Member, Supplier, SystemSetting

    if not Member.query.filter_by(member_no='M1001').first():
        db.session.add(Member(
            member_no='M1001',
            member_name='张明',
            phone='13800010001',
            level='gold',
            points=680,
            status='active',
        ))

    if not Employee.query.filter_by(employee_no='E1001').first():
        db.session.add(Employee(
            employee_no='E1001',
            employee_name='李雪',
            position='店长',
            phone='13900010001',
            work_schedule='周一至周五 09:00-18:00',
            status='active',
        ))

    if not Supplier.query.filter_by(supplier_code='S1001').first():
        db.session.add(Supplier(
            supplier_code='S1001',
            supplier_name='湖北优鲜供应链有限公司',
            contact_person='王经理',
            phone='027-88880001',
            settlement_cycle='monthly',
            status='active',
        ))

    defaults = {
        'store.name': ('校园示范超市', '门店名称'),
        'inventory.low_stock_notice': ('enabled', '库存预警通知开关'),
        'receipt.footer': ('谢谢惠顾，欢迎下次光临', '小票页脚'),
    }
    for key, (value, description) in defaults.items():
        if not SystemSetting.query.filter_by(setting_key=key).first():
            db.session.add(SystemSetting(
                setting_key=key,
                setting_value=value,
                description=description,
            ))

    db.session.commit()


def _init_seed_data_if_empty(app):
    """在演示数据库为空时导入 SQL 种子数据。"""
    from app.models import Product

    if Product.query.count() > 0:
        return

    seed_path = os.path.join(app.config.get('BASE_DIR', Config.BASE_DIR), 'data', 'SQL', 'seed_data.sql')
    if not os.path.exists(seed_path):
        app.logger.warning('seed_data_missing path=%s', seed_path)
        return

    with open(seed_path, encoding='utf-8') as seed_file:
        seed_sql = seed_file.read()

    raw_connection = db.engine.raw_connection()
    try:
        raw_connection.executescript(seed_sql)
        raw_connection.commit()
        app.logger.info('seed_data_imported path=%s', seed_path)
    finally:
        raw_connection.close()

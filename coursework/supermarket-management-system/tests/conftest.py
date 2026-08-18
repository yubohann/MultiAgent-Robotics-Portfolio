import os

import pytest

from app import create_app, db
from app.models import Category, User


@pytest.fixture
def app(tmp_path):
    test_app = create_app({
        'TESTING': True,
        'INIT_DEFAULT_DATA': False,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        'LOG_FILE': os.path.join(tmp_path, 'test-app.log'),
        'WTF_CSRF_ENABLED': False,
    })

    with test_app.app_context():
        db.create_all()
        yield test_app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def users(app):
    with app.app_context():
        admin = User(username='admin', real_name='系统管理员', role='admin', is_active=1)
        admin.set_password('admin123')
        cashier = User(username='cashier01', real_name='收银员01', role='cashier', is_active=1)
        cashier.set_password('123456')
        db.session.add_all([admin, cashier])
        db.session.commit()
        return {'admin_id': admin.user_id, 'cashier_id': cashier.user_id}


@pytest.fixture
def default_category(app):
    with app.app_context():
        category = Category(category_id=1, category_name='食品饮料', parent_id=0, sort_order=1)
        db.session.add(category)
        db.session.commit()
        return category.category_id


def login_as(client, user_id, username='admin', role='admin'):
    with client.session_transaction() as session:
        session['user_id'] = user_id
        session['username'] = username
        session['role'] = role


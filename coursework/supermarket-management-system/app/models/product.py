from datetime import datetime

from app import db


class Product(db.Model):
    """商品表"""
    __tablename__ = 'products'

    product_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    barcode = db.Column(db.String(50), unique=True)
    product_code = db.Column(db.String(50), unique=True, nullable=False)
    product_name = db.Column(db.String(100), nullable=False)
    category_id = db.Column(db.Integer, db.ForeignKey('categories.category_id'))
    unit = db.Column(db.String(20), default='件')
    purchase_price = db.Column(db.Numeric(10, 2), default=0)
    selling_price = db.Column(db.Numeric(10, 2), nullable=False)
    min_stock = db.Column(db.Integer, default=10)
    status = db.Column(db.Integer, default=1)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, onupdate=datetime.now)

    inventory = db.relationship('Inventory', backref='product', uselist=False, lazy=True)
    inventory_logs = db.relationship('InventoryLog', backref='product', lazy=True)

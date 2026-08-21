from datetime import datetime

from app import db


class Category(db.Model):
    """分类表"""
    __tablename__ = 'categories'

    category_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    category_name = db.Column(db.String(50), nullable=False)
    parent_id = db.Column(db.Integer, default=0)
    sort_order = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.now)

    products = db.relationship('Product', backref='category', lazy=True)

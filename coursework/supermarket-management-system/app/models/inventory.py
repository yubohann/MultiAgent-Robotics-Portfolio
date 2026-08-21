from datetime import datetime

from app import db


class Inventory(db.Model):
    """库存表"""
    __tablename__ = 'inventory'

    product_id = db.Column(db.Integer, db.ForeignKey('products.product_id', ondelete='CASCADE'), primary_key=True)
    quantity = db.Column(db.Integer, nullable=False, default=0)
    last_check_time = db.Column(db.DateTime)
    updated_at = db.Column(db.DateTime, onupdate=datetime.now)

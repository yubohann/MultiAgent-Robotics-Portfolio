from datetime import datetime

from app import db


class Supplier(db.Model):
    """供应商档案表"""
    __tablename__ = 'suppliers'

    supplier_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    supplier_code = db.Column(db.String(50), unique=True, nullable=False)
    supplier_name = db.Column(db.String(120), nullable=False)
    contact_person = db.Column(db.String(80))
    phone = db.Column(db.String(30))
    settlement_cycle = db.Column(db.String(30), default='monthly')
    status = db.Column(db.String(20), default='active')
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, onupdate=datetime.now)


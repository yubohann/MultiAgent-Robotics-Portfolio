from datetime import datetime

from app import db


class FinanceTransaction(db.Model):
    """财务流水（收入/支出）"""
    __tablename__ = 'finance_transactions'

    transaction_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    transaction_no = db.Column(db.String(50), unique=True, nullable=False)
    transaction_type = db.Column(db.String(20), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    amount = db.Column(db.Numeric(10, 2), nullable=False)
    payment_method = db.Column(db.String(20))
    related_order_no = db.Column(db.String(50))
    description = db.Column(db.String(300))
    occurred_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    operator_id = db.Column(db.Integer, db.ForeignKey('users.user_id'))
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    operator = db.relationship('User', lazy=True)

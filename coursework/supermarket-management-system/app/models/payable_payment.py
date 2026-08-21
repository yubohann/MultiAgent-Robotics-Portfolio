from datetime import datetime

from app import db


class PayablePayment(db.Model):
    """应付款支付记录"""
    __tablename__ = 'payable_payments'

    payment_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    payable_id = db.Column(db.Integer, db.ForeignKey('supplier_payables.payable_id'), nullable=False)
    amount = db.Column(db.Numeric(10, 2), nullable=False)
    payment_method = db.Column(db.String(20))
    paid_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    remark = db.Column(db.String(200))
    operator_id = db.Column(db.Integer, db.ForeignKey('users.user_id'))
    created_at = db.Column(db.DateTime, default=datetime.now)

    operator = db.relationship('User', lazy=True)

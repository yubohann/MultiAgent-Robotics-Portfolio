from datetime import datetime

from app import db


class SupplierPayable(db.Model):
    """供应商应付款"""
    __tablename__ = 'supplier_payables'

    payable_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    supplier_name = db.Column(db.String(100), nullable=False)
    bill_no = db.Column(db.String(60), unique=True)
    total_amount = db.Column(db.Numeric(10, 2), nullable=False)
    paid_amount = db.Column(db.Numeric(10, 2), nullable=False, default=0)
    due_date = db.Column(db.Date, nullable=False)
    status = db.Column(db.String(20), nullable=False, default='unpaid')
    note = db.Column(db.String(300))
    created_by = db.Column(db.Integer, db.ForeignKey('users.user_id'))
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    creator = db.relationship('User', lazy=True)
    payments = db.relationship(
        'PayablePayment',
        backref='payable',
        lazy=True,
        cascade='all, delete-orphan',
    )

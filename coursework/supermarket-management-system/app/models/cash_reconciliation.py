from datetime import datetime

from app import db


class CashReconciliation(db.Model):
    """日结对账记录"""
    __tablename__ = 'cash_reconciliations'
    __table_args__ = (
        db.UniqueConstraint('reconcile_date', 'payment_method', name='uq_cash_reconcile_date_method'),
    )

    reconciliation_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    reconcile_date = db.Column(db.Date, nullable=False)
    payment_method = db.Column(db.String(20), nullable=False)
    expected_amount = db.Column(db.Numeric(10, 2), nullable=False, default=0)
    actual_amount = db.Column(db.Numeric(10, 2), nullable=False, default=0)
    difference_amount = db.Column(db.Numeric(10, 2), nullable=False, default=0)
    note = db.Column(db.String(300))
    created_by = db.Column(db.Integer, db.ForeignKey('users.user_id'))
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    creator = db.relationship('User', lazy=True)

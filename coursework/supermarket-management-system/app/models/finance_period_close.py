from datetime import datetime

from app import db


class FinancePeriodClose(db.Model):
    """月度财务关账快照"""
    __tablename__ = 'finance_period_closings'

    close_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    period_month = db.Column(db.String(7), unique=True, nullable=False)
    total_sales = db.Column(db.Numeric(10, 2), nullable=False, default=0)
    other_income = db.Column(db.Numeric(10, 2), nullable=False, default=0)
    expense_amount = db.Column(db.Numeric(10, 2), nullable=False, default=0)
    gross_profit = db.Column(db.Numeric(10, 2), nullable=False, default=0)
    net_profit = db.Column(db.Numeric(10, 2), nullable=False, default=0)
    cash_inflow = db.Column(db.Numeric(10, 2), nullable=False, default=0)
    cash_outflow = db.Column(db.Numeric(10, 2), nullable=False, default=0)
    note = db.Column(db.String(300))
    closed_by = db.Column(db.Integer, db.ForeignKey('users.user_id'))
    closed_at = db.Column(db.DateTime, default=datetime.now)

    closer = db.relationship('User', lazy=True)
